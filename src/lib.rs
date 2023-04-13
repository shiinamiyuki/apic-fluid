#![allow(non_snake_case)]
pub use luisa::prelude::*;
pub use luisa::{derive::*, lang::*, macros::*, math::*, Buffer, Context, Device, Kernel};
pub use luisa_compute as luisa;
pub use luisa::glam;
pub mod cpp_extra;
pub mod fluid;
pub mod grid;
pub mod pcgsolver;
pub mod reconstruction;
pub fn trilinear_weight(off: Expr<Float3>, dim: usize) -> Expr<f32> {
    let one_minus_off = 1.0 - off.abs();
    let one_minus_off = one_minus_off.max(0.0);
    match dim {
        2 => one_minus_off.xy().reduce_prod(),
        3 => one_minus_off.reduce_prod(),
        _ => unreachable!(),
    }
}

pub fn grad_trilinear_weight(off: Expr<Float3>, dim: usize) -> Expr<Float3> {
    let sgn = Float3Expr::select(
        off.cmpgt(0.0),
        make_float3(1.0, 1.0, 1.0),
        Float3Expr::select(
            off.cmpeq(0.0),
            make_float3(0.0, 0.0, 0.0),
            -make_float3(1.0, 1.0, 1.0),
        ),
    );
    let one_minus_off = 1.0 - off.abs();
    let sgn = Float3Expr::select(one_minus_off.cmplt(0.0), Float3Expr::zero(), sgn);
    let one_minus_off = one_minus_off.max(0.0);
    match dim {
        2 => {
            let dx = -sgn.x() * one_minus_off.y();
            let dy = -sgn.y() * one_minus_off.x();
            make_float3(dx, dy, 0.0)
        }
        3 => {
            let dx = -sgn.x() * one_minus_off.yz().reduce_prod();
            let dy = -sgn.y() * one_minus_off.xz().reduce_prod();
            let dz = -sgn.z() * one_minus_off.xy().reduce_prod();
            make_float3(dx, dy, dz)
        }
        _ => unreachable!(),
    }
}
