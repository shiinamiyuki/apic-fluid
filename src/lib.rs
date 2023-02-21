#![allow(non_snake_case)]
pub use luisa::prelude::{poly::Polymorphic, *};
pub use luisa_compute as luisa;
use rayon::prelude::*;
pub mod fluid;
pub mod sparse;

pub struct Grid<T: Value> {
    pub dimension: usize,
    pub values: Buffer<T>,
    pub res: [u32; 3],
}

impl<T: Value> Grid<T> {
    pub fn new(device: Device, res: [u32; 3], dimension: usize) -> Self {
        if dimension == 2 {
            assert!(res[2] == 1);
        }
        let values = device
            .create_buffer::<T>((res[0] * res[1] * res[2]) as usize * dimension)
            .unwrap();
        Self {
            dimension,
            values,
            res,
        }
    }
    pub fn linear_index(&self, p: Expr<UVec3>) -> Expr<u32> {
        if self.dimension == 2 {
            p.x() + p.y() * const_(self.res[0])
        } else {
            p.x() + p.y() * const_(self.res[0]) + p.z() * const_(self.res[0] * self.res[1])
        }
    }
}
