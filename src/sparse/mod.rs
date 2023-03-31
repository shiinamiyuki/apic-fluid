use std::{cmp::Ordering, collections::HashMap, env::current_exe};

use parking_lot::RwLock;

use crate::{
    sparse::solver::{DiagJacobiPreconditioner, IdentityPreconditioner, PcgSolver},
    *,
};
pub mod solver;

#[derive(Clone, Copy)]
pub struct Triplet {
    pub row: u32,
    pub col: u32,
    pub value: f32,
}
impl Triplet {
    pub fn new(row: u32, col: u32, value: f32) -> Self {
        Self { row, col, value }
    }
}
impl PartialEq for Triplet {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}
impl Eq for Triplet {}
impl PartialOrd for Triplet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.row.cmp(&other.row).then(self.col.cmp(&other.col)))
    }
}
impl Ord for Triplet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct SparseKernels {
    device: Device,
    spmv: Kernel<(SparseMatrixData, Buffer<f32>, Buffer<f32>)>,
    spmv_sub: Kernel<(SparseMatrixData, Buffer<f32>, Buffer<f32>)>,
    diag: Kernel<(SparseMatrixData, Buffer<f32>)>,
    vec_add_scaled: Kernel<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>,
    zero: Kernel<(Buffer<f32>,)>,
}
#[derive(KernelArg)]
pub struct SparseMatrixData {
    values: Buffer<f32>,
    col_indices: Buffer<u32>,
    row_offsets: Buffer<u32>,
}

impl SparseMatrixData {
    fn from_triplets(device: Device, n: usize, mut triplets: Vec<Triplet>) -> Self {
        triplets.par_sort();
        let mut values = vec![];
        let mut col_indices = vec![];
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut prev_col = -1i32;
        let mut prev_row = -1i32;
        for Triplet { row, col, value } in triplets {
            while row >= row_offsets.len() as u32 {
                row_offsets.push(col_indices.len() as u32);
            }
            if col as i32 != prev_col {
                col_indices.push(col);
            }
            if prev_row == row as i32 && col as i32 == prev_col {
                *values.last_mut().unwrap() += value;
            } else {
                values.push(value);
            }
            prev_col = col as i32;
            prev_row = row as i32;
        }
        row_offsets.push(col_indices.len() as u32);
        assert_eq!(row_offsets.len(), n + 1);
        Self {
            values: device.create_buffer_from_slice(&values).unwrap(),
            col_indices: device.create_buffer_from_slice(&col_indices).unwrap(),
            row_offsets: device.create_buffer_from_slice(&row_offsets).unwrap(),
        }
    }
}
pub struct SparseMatrix {
    pub(crate) data: SparseMatrixData,
    pub(crate) device: Device,
    pub(crate) n: usize,
}

pub struct Vector {
    pub(crate) data: Buffer<f32>,
    pub(crate) device: Device,
    pub(crate) n: usize,
}
impl Vector {
    pub fn values(&self) -> &Buffer<f32> {
        &self.data
    }
    pub fn new(device: Device, n: usize) -> Self {
        Self {
            data: device.create_buffer_from_fn::<f32>(n, |_| 0.0).unwrap(),
            device,
            n,
        }
    }
    pub fn from_slice(device: Device, slice: &[f32]) -> Self {
        Self {
            data: device.create_buffer_from_slice(slice).unwrap(),
            device,
            n: slice.len(),
        }
    }
    pub fn zero(&self) {
        SparseKernels::with(self.device.clone(), |kernels| {
            kernels
                .zero
                .dispatch([self.n as u32, 1, 1], &self.data)
                .unwrap();
        });
    }
    pub fn dot(&self, b: &Vector) -> f32 {
        let a = self.values().copy_to_vec();
        let b = b.values().copy_to_vec();
        a.par_iter().zip(b.par_iter()).map(|(a, b)| a * b).sum()
    }
    pub fn abs_max(&self) -> f32 {
        let a = self.values().copy_to_vec();
        // dbg!(&a);
        a.par_iter()
            .map(|a| a.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
    pub fn add_scaled(&self, k: f32, v: &Vector) {
        let k = self.device.create_buffer_from_slice(&[k]).unwrap();
        SparseKernels::with(self.device.clone(), |kernels| {
            kernels
                .vec_add_scaled
                .dispatch([self.n as u32, 1, 1], &k, &v.data, &self.data)
                .unwrap();
        });
    }
    pub fn to_numpy(&self, var: &str) -> String {
        let a = self.values().copy_to_vec();
        format!("{} = np.array({:?}, dtype=np.float32)", var, a)
    }
}

impl SparseMatrix {
    // out = A * v
    pub fn multiply(&self, v: &Vector, out: &Vector) {
        SparseKernels::with(self.device.clone(), |kernels| {
            kernels
                .spmv
                .dispatch([self.n as u32, 1, 1], &self.data, &v.data, &out.data)
                .unwrap();
        });
    }
    // out = out - A * v
    pub fn multiply_sub(&self, v: &Vector, out: &Vector) {
        SparseKernels::with(self.device.clone(), |kernels| {
            kernels
                .spmv_sub
                .dispatch([self.n as u32, 1, 1], &self.data, &v.data, &out.data)
                .unwrap();
        });
    }
    pub fn from_triplets(device: Device, n: usize, triplets: Vec<Triplet>) -> Self {
        let data = SparseMatrixData::from_triplets(device.clone(), n, triplets);
        Self { data, n, device }
    }
    pub fn to_triplets(&self) -> Vec<Triplet> {
        let values = self.data.values.copy_to_vec();
        let col_indices = self.data.col_indices.copy_to_vec();
        let row_offsets = self.data.row_offsets.copy_to_vec();
        let mut out = vec![];
        for i in 0..self.n {
            for j in row_offsets[i] as usize..row_offsets[i + 1] as usize {
                out.push(Triplet {
                    row: i as u32,
                    col: col_indices[j],
                    value: values[j],
                })
            }
        }
        out
    }
    pub fn diagonal(&self) -> Vector {
        let out = Vector::new(self.device.clone(), self.n);
        SparseKernels::with(self.device.clone(), |kernels| {
            kernels
                .diag
                .dispatch([self.n as u32, 1, 1], &self.data, &out.data)
                .unwrap();
        });
        out
    }
    pub fn transpose(&self) -> Self {
        let mut triplets = self.to_triplets();
        triplets
            .par_iter_mut()
            .for_each(|t| std::mem::swap(&mut t.row, &mut t.col));
        Self::from_triplets(self.device.clone(), self.n, triplets)
    }
    pub fn identity(device: Device, n: usize) -> Self {
        let mut triplets = vec![];
        for i in 0..n {
            triplets.push(Triplet {
                row: i as u32,
                col: i as u32,
                value: 1.0,
            });
        }
        Self::from_triplets(device, n, triplets)
    }
    pub fn to_numpy(&self, var: &str) -> String {
        let values = self.data.values.copy_to_vec();
        let col_indices = self.data.col_indices.copy_to_vec();
        let row_offsets = self.data.row_offsets.copy_to_vec();
        let mut out = String::new();
        use std::fmt::Write;
        writeln!(
            out,
            "values = np.array([{}])",
            values
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(
            out,
            "col_indices = np.array([{}])",
            col_indices
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(
            out,
            "row_offsets = np.array([{}])",
            row_offsets
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(
            out,
            "{} = csr_array((values, col_indices, row_offsets), shape=({},{}))",
            var, self.n, self.n
        )
        .unwrap();
        out
    }
}

impl SparseKernels {
    pub fn new(device: Device) -> Self {
        let spmv = device
            .create_kernel::<(SparseMatrixData, Buffer<f32>, Buffer<f32>)>(
                &|a: SparseMatrixDataVar, v: BufferVar<f32>, out: BufferVar<f32>| {
                    let r = dispatch_id().x();
                    let sum = var!(f32);
                    let j = var!(u32, a.row_offsets.read(r));
                    let end = a.row_offsets.read(r + 1);
                    while_!(j.load().cmplt(end), {
                        let i = a.col_indices.read(j.load());
                        sum.store(sum.load() + a.values.read(j.load()) * v.read(i));
                        j.store(j.load() + 1u32);
                    });
                    out.write(r, sum.load());
                },
            )
            .unwrap();
        let spmv_sub = device
            .create_kernel::<(SparseMatrixData, Buffer<f32>, Buffer<f32>)>(
                &|a: SparseMatrixDataVar, v: BufferVar<f32>, out: BufferVar<f32>| {
                    let r = dispatch_id().x();
                    let sum = var!(f32);
                    let j = var!(u32, a.row_offsets.read(r));
                    let end = a.row_offsets.read(r + 1);
                    while_!(j.load().cmplt(end), {
                        let i = a.col_indices.read(j.load());
                        sum.store(sum.load() + a.values.read(j.load()) * v.read(i));
                        j.store(j.load() + 1u32);
                    });
                    out.write(r, out.read(r) - sum.load());
                },
            )
            .unwrap();
        let vec_add_scaled = device
            .create_kernel::<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                &|a: BufferVar<f32>, b: BufferVar<f32>, out: BufferVar<f32>| {
                    let i = dispatch_id().x();
                    out.write(i, out.read(i) + a.read(0) * b.read(i));
                },
            )
            .unwrap();
        let zero = device
            .create_kernel::<(Buffer<f32>,)>(&|a: BufferVar<f32>| {
                let i = dispatch_id().x();
                a.write(i, 0.0);
            })
            .unwrap();
        let diag = device
            .create_kernel::<(SparseMatrixData, Buffer<f32>)>(&|a: SparseMatrixDataVar, d| {
                let r = dispatch_id().x();
                let v = var!(f32);
                let j = var!(u32, a.row_offsets.read(r));
                let end = a.row_offsets.read(r + 1);
                while_!(j.load().cmplt(end), {
                    let i = a.col_indices.read(j.load());
                    if_!(i.cmpeq(r), {
                        v.store(a.values.read(j.load()));
                        break_();
                    });
                    j.store(j.load() + 1u32);
                });
                d.write(r, v.load());
            })
            .unwrap();
        Self {
            device,
            spmv,
            spmv_sub,
            vec_add_scaled,
            zero,
            diag,
        }
    }
    pub fn with(device: Device, mut f: impl FnMut(&SparseKernels)) {
        use lazy_static::lazy_static;
        lazy_static! {
            static ref KERNELS: RwLock<HashMap<Device, SparseKernels>> =
                RwLock::new(HashMap::new());
        }
        let kernels = KERNELS.read();
        if let Some(kernels) = kernels.get(&device) {
            f(kernels);
        } else {
            drop(kernels);
            let kernels = SparseKernels::new(device.clone());
            KERNELS.write().insert(device.clone(), kernels);
            f(&KERNELS.read()[&device]);
        }
    }
}
#[allow(non_snake_case)]
pub fn test_sparse() {
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_cpu_device().unwrap();
    let mut triplets = Vec::new();
    triplets.push(Triplet::new(0, 0, 4.0));
    triplets.push(Triplet::new(1, 1, 5.0));
    triplets.push(Triplet::new(0, 2, 1.0));
    triplets.push(Triplet::new(2, 0, 1.0));
    triplets.push(Triplet::new(2, 2, 3.0));
    triplets.push(Triplet::new(3, 3, 4.0));
    triplets.push(Triplet::new(3, 2, 2.0));
    triplets.push(Triplet::new(2, 3, 2.0));
    let A = SparseMatrix::from_triplets(device.clone(), 4, triplets);
    let b = Vector::new(device.clone(), 4);
    let x = Vector::from_slice(device.clone(), &[-1.5, -0.5, -1.0, 2.0]);
    A.multiply(&x, &b);
    println!("{}", A.to_numpy("A"));
    println!("{}", x.to_numpy("x"));
    println!("{}", b.to_numpy("b"));
    let mut solver = PcgSolver::new(device.clone(), 4);
    let out = Vector::new(device.clone(), 4);
    solver
        .solve::<DiagJacobiPreconditioner>(&A, &b, &out)
        .unwrap();
    println!("{}", out.to_numpy("out"));
}
