#![allow(non_snake_case)]
pub use luisa::prelude::{poly::Polymorphic, *};
pub use luisa_compute as luisa;
use rayon::prelude::*;
pub mod fluid;
pub mod sparse;
pub mod grid;

