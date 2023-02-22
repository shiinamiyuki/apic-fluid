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
    pub shift: [T; 3],
}


impl<T: Value> Grid<T> {
    pub fn new(device: Device, res: [u32; 3], dimension: usize, shift: [T; 3]) -> Self {
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
            shift,
        }
    }
    pub fn linear_index(&self, p: Expr<UVec3>) -> Expr<u32> {
        let p = p.clamp(
            make_uint3(0, 0, 0),
            make_uint3(self.res[0] - 1, self.res[1] - 1, self.res[2] - 1),
        );
        if self.dimension == 2 {
            p.x() + p.y() * const_(self.res[0])
        } else {
            p.x() + p.y() * const_(self.res[0]) + p.z() * const_(self.res[0] * self.res[1])
        }
    }
    pub fn at_index(&self, p: Expr<UVec3>) -> Expr<T> {
        let index = self.linear_index(p);
        self.values.var().read(index)
    }
    pub fn set_index(&self, p: Expr<UVec3>, v: Expr<T>) {
        let index = self.linear_index(p);
        self.values.var().write(index, v);
    }
}

impl Grid<f32> {
    pub fn bilinear(&self, p: Expr<Vec3>) -> Expr<f32> {
        if self.dimension == 2 {
            let p = p - make_float3(self.shift[0], self.shift[1], 0.0);
            let ip = p.floor().int();
            let offset = p - ip.float();
            let v00 = self.at_index(ip.uint());
            let v01 = self.at_index(ip.uint() + make_uint3(1, 0, 0));
            let v10 = self.at_index(ip.uint() + make_uint3(0, 1, 0));
            let v11 = self.at_index(ip.uint() + make_uint3(1, 1, 0));
            let v0 = (1.0 - offset.x()) * v00 + offset.x() * v01;
            let v1 = (1.0 - offset.x()) * v10 + offset.x() * v11;
            let v = (1.0 - offset.y()) * v0 + offset.y() * v1;
            v
        } else {
            todo!()
        }
    }
    pub fn at_index_or_zero(&self, p: Expr<IVec3>) -> Expr<f32> {
        if self.dimension == 2 {
            let oob = p.xy().cmplt(IVec2Expr::zero())
                | p.xy().cmpge(make_uint2(self.res[0], self.res[1]).int());
            if_!(oob.any(), {
                const_(0.0f32)
            }, else{
                let index = self.linear_index(p.as_uvec3());
            self.values.var().read(index)
            })
        } else {
            let oob = p.cmplt(IVec3Expr::zero())
                | p.cmpge(make_uint3(self.res[0], self.res[1], self.res[2]).int());
            if_!(oob.any(), {
                const_(0.0f32)
            }, else{
                let index = self.linear_index(p.as_uvec3());
            self.values.var().read(index)
            })
        }
    }
}
