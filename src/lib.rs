#![allow(non_snake_case)]
pub use luisa::prelude::*;
pub use luisa_compute as luisa;
pub use luisa::{Context, Buffer, Kernel, Device, lang::*, macros::*, derive::*, math::*};
use rayon::prelude::*;
pub mod fluid;
pub mod sparse;
pub mod grid;

