use crate::*;
#[derive(KernelArg)]
pub struct Stencil {
    pub coeff: Buffer<f32>,
    #[luisa(exclude)]
    pub n: [u32; 3],
    pub offsets: Buffer<Int3>,
}

impl Stencil {
    pub fn new(device: Device, n: [u32; 3], offsets: &[Int3]) -> Self {
        let N = n[0] * n[1] * n[2];
        let coeff = device.create_buffer((N as usize) * offsets.len()).unwrap();
        let offsets = device.create_buffer_from_slice(offsets).unwrap();
        Self { coeff, offsets, n }
    }
}
#[derive(Clone, Copy, PartialOrd, PartialEq, Debug)]
pub enum Preconditioner {
    Identity,
    IncompletePoisson,
}
pub struct PcgSolver {
    device: Device,
    pub z: Buffer<f32>,
    pub r: Buffer<f32>,
    pub s: Buffer<f32>,
    pub abs_max_tmp: Buffer<f32>,
    pub reduce_add_tmp: Buffer<f32>,
    pub max_iter: usize,
    pub tol: f32,
    apply_A: Kernel<(Stencil, Buffer<f32>, Buffer<f32>, i32)>,
    apply_preconditioner: Kernel<(Stencil, Buffer<f32>, Buffer<f32>)>,
    dot: Kernel<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>,
    reduce_abs_max: Kernel<(Buffer<f32>, Buffer<f32>)>,
    zero: Kernel<(Buffer<f32>,)>,
    add_scaled: Kernel<(f32, Buffer<f32>, Buffer<f32>)>,
    n: [u32; 3],
}
impl PcgSolver {
    const PAR_REDUCE_BLOCK_SIZE: usize = 1024;
    pub fn new(
        device: Device,
        n: [u32; 3],
        max_iter: usize,
        tol: f32,
        precond: Preconditioner,
    ) -> Self {
        let N = n[0] * n[1] * n[2];
        let z = device.create_buffer(N as usize).unwrap();
        let r = device.create_buffer(N as usize).unwrap();
        let s = device.create_buffer(N as usize).unwrap();
        let reduce_add_tmp = device.create_buffer(Self::PAR_REDUCE_BLOCK_SIZE).unwrap();
        let abs_max_tmp = device.create_buffer(Self::PAR_REDUCE_BLOCK_SIZE).unwrap();
        let apply_A = device
            .create_kernel_async::<(Stencil, Buffer<f32>, Buffer<f32>, i32)>(
                &|A: StencilVar, x: BufferVar<f32>, out: BufferVar<f32>, sub: Expr<i32>| {
                    let i = dispatch_id().x();
                    set_block_size([512, 1, 1]);
                    let ix = i % n[0];
                    let iy = (i / n[0]) % n[1];
                    let iz = i / (n[0] * n[1]);
                    let noffsets = A.offsets.len();
                    let sum = var!(f32, x.read(i) * A.coeff.read(i * noffsets));
                    let c = var!(u32, 1);
                    while_!(c.load().cmplt(noffsets), {
                        let off = A.offsets.read(c.load());
                        let p = make_uint3(ix, iy, iz).int() + off;
                        if_!(
                            p.cmpge(0).all() & p.cmplt(make_uint3(n[0], n[1], n[2]).int()).all(),
                            {
                                let p = p.uint();
                                let ip = p.x() + p.y() * n[0] + p.z() * n[0] * n[1];
                                sum.store(
                                    sum.load() + x.read(ip) * A.coeff.read(i * noffsets + c.load()),
                                );
                            }
                        );
                        c.store(c.load() + 1);
                    });
                    if_!(sub.cmpeq(0), {
                        out.write(i, sum.load());
                    }, else{
                        out.write(i, out.read(i) - sum.load());
                    });
                },
            )
            .unwrap();
        let apply_preconditioner = device
            .create_kernel_async::<(Stencil, Buffer<f32>, Buffer<f32>)>(
                &|A: StencilVar, x: BufferVar<f32>, out: BufferVar<f32>| {
                    let i = dispatch_id().x();
                    let ix = i % n[0];
                    let iy = (i / n[0]) % n[1];
                    let iz = i / (n[0] * n[1]);
                    let noffsets = A.offsets.len();
                    set_block_size([512, 1, 1]);
                    match precond {
                        Preconditioner::Identity => {
                            out.write(i, x.read(i));
                        }
                        Preconditioner::IncompletePoisson => {
                            let sum = var!(f32, 0.0);
                            let m = var!(f32, 1.0);
                            let c = var!(u32, 1);
                            while_!(c.load().cmplt(noffsets), {
                                let off = A.offsets.read(c.load());
                                let p = make_uint3(ix, iy, iz).int() + off;
                                if_!(
                                    p.cmpge(0).all()
                                        & p.cmplt(make_uint3(n[0], n[1], n[2]).int()).all(),
                                    {
                                        let p = p.uint();
                                        let ip = p.x() + p.y() * n[0] + p.z() * n[0] * n[1];
                                        let mp = 1.0 / A.coeff.read(ip * noffsets);
                                        sum.store(sum.load() + x.read(ip) * mp);
                                        if_!((c.load() % 1).cmpeq(0), {
                                            m.store(m.load() + mp * mp);
                                        });
                                    }
                                );
                                // m.store()

                                c.store(c.load() + 1);
                            });
                            // cpu_dbg!(m.load());
                            sum.store(sum.load() + x.read(i) * m.load());
                            out.write(i, sum.load());
                        }
                    }
                },
            )
            .unwrap();
        let dot =
            device
                .create_kernel_async::<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                    &|x: BufferVar<f32>, y: BufferVar<f32>, out: BufferVar<f32>| {
                        let i = dispatch_id().x();
                        set_block_size([Self::PAR_REDUCE_BLOCK_SIZE as u32 + 1, 1, 1]);
                        out.atomic_fetch_add(
                            i % Self::PAR_REDUCE_BLOCK_SIZE as u32,
                            x.read(i) * y.read(i),
                        );
                    },
                )
                .unwrap();
        let reduce_abs_max = device
            .create_kernel_async::<(Buffer<f32>, Buffer<f32>)>(
                &|x: BufferVar<f32>, out: BufferVar<f32>| {
                    let i = dispatch_id().x();
                    set_block_size([Self::PAR_REDUCE_BLOCK_SIZE as u32 + 1, 1, 1]);
                    out.atomic_fetch_max(i % Self::PAR_REDUCE_BLOCK_SIZE as u32, x.read(i).abs());
                },
            )
            .unwrap();
        let zero = device
            .create_kernel_async::<(Buffer<f32>,)>(&|x: BufferVar<f32>| {
                let i = dispatch_id().x();
                x.write(i, 0.0);
            })
            .unwrap();
        let add_scaled = device
            .create_kernel_async::<(f32, Buffer<f32>, Buffer<f32>)>(
                &|a: Expr<f32>, x: BufferVar<f32>, out: BufferVar<f32>| {
                    let i = dispatch_id().x();
                    out.write(i, out.read(i) + a * x.read(i));
                },
            )
            .unwrap();
        Self {
            device,
            z,
            r,
            s,
            max_iter,
            tol,
            n,
            apply_A,
            apply_preconditioner,
            dot,
            abs_max_tmp,
            reduce_add_tmp,
            reduce_abs_max,
            zero,
            add_scaled,
        }
    }
    pub fn dot(&self, x: &Buffer<f32>, y: &Buffer<f32>) -> f32 {
        let s = self.device.default_stream();
        s.with_scope(|s| {
            s.submit([
                self.zero.dispatch_async(
                    [Self::PAR_REDUCE_BLOCK_SIZE as u32, 1, 1],
                    &self.reduce_add_tmp,
                ),
                self.dot
                    .dispatch_async([x.len() as u32, 1, 1], x, y, &self.reduce_add_tmp),
            ])
            .unwrap();
        });
        self.reduce_add_tmp.copy_to_vec().iter().sum()
    }
    pub fn abs_max(&self, x: &Buffer<f32>) -> f32 {
        let s = self.device.default_stream();
        s.with_scope(|s| {
            s.submit([
                self.zero.dispatch_async(
                    [Self::PAR_REDUCE_BLOCK_SIZE as u32, 1, 1],
                    &self.abs_max_tmp,
                ),
                self.reduce_abs_max
                    .dispatch_async([x.len() as u32, 1, 1], x, &self.abs_max_tmp),
            ])
            .unwrap();
        });
        self.abs_max_tmp
            .copy_to_vec()
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
    pub fn apply_A(&self, A: &Stencil, x: &Buffer<f32>, out: &Buffer<f32>, sub: bool) {
        self.apply_A
            .dispatch(
                [x.len() as u32, 1, 1],
                A,
                x,
                out,
                &if sub { 1i32 } else { 0i32 },
            )
            .unwrap();
    }
    pub fn apply_preconditioner(&self, A: &Stencil, x: &Buffer<f32>, out: &Buffer<f32>) {
        self.apply_preconditioner
            .dispatch([x.len() as u32, 1, 1], A, x, out)
            .unwrap();
    }
    fn add_scaled(&self, k: f32, x: &Buffer<f32>, out: &Buffer<f32>) {
        self.add_scaled
            .dispatch([x.len() as u32, 1, 1], &k, x, out)
            .unwrap();
    }
    pub fn solve(&self, A: &Stencil, b: &Buffer<f32>, x: &Buffer<f32>) -> Option<usize> {
        assert_eq!(A.n, self.n);
        b.copy_to_buffer(&self.r);
        self.apply_A(A, x, &self.r, true);
        let residual = self.abs_max(&self.r);
        if residual < self.tol {
            return Some(0);
        }
        let tol = self.tol * residual;
        self.apply_preconditioner(A, &self.r, &self.z);
        let mut rho = self.dot(&self.r, &self.z);
        if rho == 0.0 || rho.is_nan() {
            return None;
        }
        self.z.copy_to_buffer(&self.s);
        for i in 0..self.max_iter {
            self.apply_A(A, &self.s, &self.z, false);
            let alpha = rho / self.dot(&self.s, &self.z);
            self.add_scaled(alpha, &self.s, &x);
            self.add_scaled(-alpha, &self.z, &self.r);
            let residual = self.abs_max(&self.r);
            dbg!(i, residual);
            if residual < tol {
                return Some(i + 1);
            }
            self.apply_preconditioner(A, &self.r, &self.z);
            let rho_new = self.dot(&self.r, &self.z);
            let beta = rho_new / rho;
            self.add_scaled(beta, &self.s, &self.z);
            rho = rho_new;
            self.z.copy_to_buffer(&self.s);
        }
        None
    }
}
#[link(name = "solve")]
extern "C" {
    pub fn eigen_pcg_solve(
        nx: i32,
        ny: i32,
        nz: i32,
        stencil: *const f32,
        offsets: *const i32,
        noffsets: i32,
        b: *const f32,
        out: *mut f32,
    );
}
pub fn eigen_solve(stencil: &Stencil, b: &Buffer<f32>, out: &Buffer<f32>) {
    assert_eq!(stencil.n[0] * stencil.n[1] * stencil.n[2], b.len() as u32);
    assert_eq!(stencil.n[0] * stencil.n[1] * stencil.n[2], out.len() as u32);
    let coeff = stencil.coeff.copy_to_vec();
    let offsets = stencil.offsets.copy_to_vec();
    let offsets = offsets.iter().map(|x| [x.x, x.y, x.z]).collect::<Vec<_>>();
    let b = b.copy_to_vec();
    let mut out_ = out.copy_to_vec();
    unsafe {
        eigen_pcg_solve(
            stencil.n[0] as i32,
            stencil.n[1] as i32,
            stencil.n[2] as i32,
            coeff.as_ptr(),
            offsets.as_ptr() as *const i32,
            offsets.len() as i32,
            b.as_ptr(),
            out_.as_mut_ptr() as *mut f32,
        );
    }
    out.copy_from(&out_);
}
