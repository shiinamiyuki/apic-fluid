use crate::*;

use super::{SparseMatrix, Vector};
#[allow(non_snake_case)]
pub struct PcgSolver {
    r: Vector,
    z: Vector,
    n: usize,
    s: Vector,
    Pinv: SparseMatrix,
    max_iterations: usize,
    tol: f32,
}

impl PcgSolver {
    pub fn apply_preconditioner(&self, x: &Vector, y: &Vector) {
        assert_eq!(x.n, self.n);
        assert_eq!(y.n, self.n);
        self.Pinv.multiply(&x, &y);
    }
    #[allow(non_snake_case)]
    pub fn solve(&mut self, A: &SparseMatrix, b: &Vector, x: &Vector) -> Option<usize> {
        assert_eq!(A.n, self.n);
        assert_eq!(b.n, self.n);
        b.data.copy_to_buffer(&self.r.data);
        A.multiply_sub(&x, &self.r);
        let residual = self.r.abs_max();
        if residual < self.tol {
            return Some(0);
        }
        self.apply_preconditioner(&self.r, &self.z);
        let mut rho = self.r.dot(&self.z);
        if rho == 0.0 || rho.is_nan() {
            return None;
        }
        for i in 0..self.max_iterations {
            A.multiply(&self.s, &self.z);
            let alpha = rho / self.s.dot(&self.z);
            x.add_scaled(alpha, &self.s);
            self.r.add_scaled(-alpha, &self.z);
            if residual < self.tol {
                return Some(i + 1);
            }
            self.apply_preconditioner(&self.r, &self.z);
            let rho_new = self.r.dot(&self.z);
            let beta = rho_new / rho;
            self.z.add_scaled(beta, &self.s);
            std::mem::swap(&mut self.s, &mut self.z);
            rho = rho_new;
        }
        None
    }
}
