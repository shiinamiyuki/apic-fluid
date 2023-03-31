use crate::*;
pub struct Stencil {
    pub coeff: Buffer<f32>,
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

// pub struct PcgSolver {
//     pub stencil: Stencil,
//     pub z: Buffer<f32>,
//     pub r: Buffer<f32>,
//     pub s: Buffer<f32>,
// }
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
    let offsets = offsets
        .iter()
        .map(|x| [x.x, x.y, x.z])
        .collect::<Vec<_>>();
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
