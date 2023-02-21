use crate::*;

use super::{SparseMatrix, Triplet, Vector};

pub trait Preconditioner {
    fn precompute(A: &SparseMatrix) -> Self;
    fn apply(&self, x: &Vector, out: &Vector);
}
pub struct IdentityPreconditioner;

pub struct DiagJacobiPreconditioner {
    diag: Vector,
    div: Kernel<(Buffer<f32>,)>,
}
impl Preconditioner for DiagJacobiPreconditioner {
    fn precompute(A: &SparseMatrix) -> Self {
        let diag = A.diagonal();
        let device = A.device.clone();
        let div = device
            .create_kernel::<(Buffer<f32>,)>(&|a| {
                let tid = dispatch_id().x();
                a.write(tid, a.read(tid) / diag.data.var().read(tid));
            })
            .unwrap();
        Self { diag, div }
    }
    fn apply(&self, x: &Vector, out: &Vector) {
        x.data.copy_to_buffer(&out.data);
        self.div
            .dispatch([self.diag.n as u32, 1, 1], &out.data)
            .unwrap();
    }
}
impl Preconditioner for IdentityPreconditioner {
    fn precompute(_A: &SparseMatrix) -> Self {
        Self
    }
    fn apply(&self, x: &Vector, out: &Vector) {
        x.data.copy_to_buffer(&out.data);
    }
}

pub struct PcgSolver {
    r: Vector,
    z: Vector,
    n: usize,
    s: Vector,
    max_iterations: usize,
    tol: f32,
}

impl PcgSolver {
    #[allow(non_snake_case)]
    pub fn solve<P: Preconditioner>(
        &mut self,
        A: &SparseMatrix,
        b: &Vector,
        x: &Vector,
    ) -> Option<usize> {
        assert_eq!(A.n, self.n);
        assert_eq!(b.n, self.n);
        b.data.copy_to_buffer(&self.r.data);
        A.multiply_sub(&x, &self.r);
        let residual = self.r.abs_max();
        if residual < self.tol {
            return Some(0);
        }
        let precond = P::precompute(A);
        precond.apply(&self.r, &self.z);
        println!("{}", self.r.to_numpy("r"));
        println!("{}", self.z.to_numpy("z"));
        let mut rho = self.r.dot(&self.z);
        if rho == 0.0 || rho.is_nan() {
            return None;
        }
        self.z.data.copy_to_buffer(&self.s.data);
        // dbg!(rho);
        for i in 0..self.max_iterations {
            A.multiply(&self.s, &self.z);
            let alpha = rho / self.s.dot(&self.z);
            // dbg!(alpha);
            x.add_scaled(alpha, &self.s);
            self.r.add_scaled(-alpha, &self.z);
            let residual = self.r.abs_max();
            dbg!(residual);
            if residual < self.tol {
                return Some(i + 1);
            }
            precond.apply(&self.r, &self.z);
            let rho_new = self.r.dot(&self.z);
            let beta = rho_new / rho;
            self.z.add_scaled(beta, &self.s);
            std::mem::swap(&mut self.s, &mut self.z);
            rho = rho_new;
        }
        None
    }
    pub fn new(device: Device, n: usize) -> Self {
        Self {
            r: Vector::new(device.clone(), n),
            z: Vector::new(device.clone(), n),
            n,
            s: Vector::new(device.clone(), n),
            max_iterations: n.min(100),
            tol: 1e-6,
        }
    }
}
