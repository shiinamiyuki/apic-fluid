#![allow(non_snake_case)]
pub use luisa::prelude::*;
pub use luisa::{derive::*, lang::*, macros::*, math::*, Buffer, Context, Device, Kernel};
pub use luisa_compute as luisa;
use rayon::prelude::*;
pub mod fluid;
pub mod grid;
pub mod pcgsolver;

pub fn trilinear_weight(off: Expr<Float3>, dim: usize) -> Expr<f32> {
    match dim {
        2 => (1.0 - off.abs()).xy().reduce_prod(),
        3 => (1.0 - off.abs()).xyz().reduce_prod(),
        _ => unreachable!(),
    }
}

pub fn grad_trilinear_weight(off: Expr<Float3>, dim: usize) -> Expr<Float3> {
    let one_minus_off = 1.0 - off.abs();

    match dim {
        2 => {
            let dx = one_minus_off.y();
            let dy = one_minus_off.x();
            make_float3(dx, dy, 0.0)
        }
        3 => {
            let dx = one_minus_off.yz().reduce_prod();
            let dy = one_minus_off.xz().reduce_prod();
            let dz = one_minus_off.xy().reduce_prod();
            make_float3(dx, dy, dz)
        }
        _ => unreachable!(),
    }
}
