use crate::*;
pub struct Stencil {
    pub coeff: Buffer<f32>,
    pub offsets: Buffer<i32>,
}

impl Stencil {
    pub fn new(device: Device, n: usize, offsets: &[i32]) -> Self {
        let coeff = device.create_buffer(n * offsets.len()).unwrap();
        let offsets = device.create_buffer_from_slice(offsets).unwrap();
        Self { coeff, offsets }
    }
}

// pub struct PcgSolver {
//     pub stencil: Stencil,
//     pub z: Buffer<f32>,
//     pub r: Buffer<f32>,
//     pub s: Buffer<f32>,
// }

extern "C" {
    pub fn eigen_pcg_solve(
        n: i32,
        stencil: *const f32,
        offsets: *const i32,
        noffsets: i32,
        b: *const f32,
        out: *mut f32,
    );
}
pub fn eigen_solve(stencil: &Stencil, b: &Buffer<f32>, out: &Buffer<f32>) {
    let coeff = stencil.coeff.copy_to_vec();
    let offsets = stencil.offsets.copy_to_vec();
    let b = b.copy_to_vec();
    let mut out_ = out.copy_to_vec();
    unsafe {
        eigen_pcg_solve(
            b.len() as i32,
            coeff.as_ptr(),
            offsets.as_ptr(),
            offsets.len() as i32,
            b.as_ptr(),
            out_.as_mut_ptr() as *mut f32,
        );
    }
}
