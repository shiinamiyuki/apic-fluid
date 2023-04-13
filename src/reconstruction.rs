use std::ffi::CString;

use crate::fluid::Particle;
use crate::grid::Grid;
use crate::*;
#[allow(dead_code)]
pub struct AnisotropicDiffusion {
    res: [u32; 3],
    device: Device,
    phi: Grid<f32>,
    GV: Vec<[f32; 3]>,
    search: Grid<bool>,
    G: Buffer<Mat3>,
    G_norm: Buffer<f32>,
    x_bar: Buffer<Float3>,
    build_particle_list_kernel: Kernel<(Buffer<Particle>,)>,
    diffusion_kernel: Kernel<(Buffer<Particle>,)>,
    compute_x_bar_kernel: Kernel<(Buffer<Particle>,)>,
    compute_G_kernel: Kernel<(Buffer<Particle>,)>,
    h: f32,
}
fn w(d: Expr<Float3>, r: Expr<f32>) -> Expr<f32> {
    let k = d.length() / r;
    let k3 = k * k * k;
    (1.0 - k3).max(0.0)
}
#[repr(C)]
#[derive(Clone, Copy, Value, Debug)]
struct ComputeGData {
    C: Mat3,
    G: Mat3,
    h: f32,
    N: u32,
    G_norm: f32,
}

impl AnisotropicDiffusion {
    pub fn new(device: Device, res: [u32; 3], h: f32, radius: f32, count: usize) -> Self {
        let mut search = Grid::new(
            device.clone(),
            [res[0] + 2, res[1] + 2, res[2] + 2],
            3,
            [-h, -h, -h],
            h,
        );
        search.init_particle_list(count);
        //  let phi = Grid::new(
        //     device.clone(),
        //     [res[0] * 8 + 16, res[1] * 8 + 16, res[2] * 8 + 16],
        //     3,
        //     [-h, -h, -h],
        //     h * 0.125,
        // );
        // let GV = Grid::<Float3>::new(
        //     device.clone(),
        //     [res[0] * 8 + 16, res[1] * 8 + 16, res[2] * 8 + 16],
        //     3,
        //     [-h, -h, -h],
        //     h * 0.125,
        // );

        let phi = Grid::new(
            device.clone(),
            [res[0] * 4 + 8, res[1] * 4 + 8, res[2] * 4 + 8],
            3,
            [-h, -h, -h],
            h * 0.25,
        );
        let GV = Grid::<Float3>::new(
            device.clone(),
            [res[0] * 4 + 8, res[1] * 4 + 8, res[2] * 4 + 8],
            3,
            [-h, -h, -h],
            h * 0.25,
        );
        // let phi = Grid::new(
        //     device.clone(),
        //     [res[0] * 2 + 4, res[1] * 2 + 4, res[2] * 2 + 4],
        //     3,
        //     [-h, -h, -h],
        //     h * 0.5,
        // );
        // let GV = Grid::<Float3>::new(
        //     device.clone(),
        //     [res[0] * 2 + 4, res[1] * 2 + 4, res[2] * 2 + 4],
        //     3,
        //     [-h, -h, -h],
        //     h * 0.5,
        // );
        device
            .create_kernel::<()>(&|| {
                let i = dispatch_id().xyz();
                let p = GV.pos_i_to_f(i.int());
                GV.set_index(i, p)
            })
            .unwrap()
            .dispatch(GV.res)
            .unwrap();
        let GV = GV
            .values
            .copy_to_vec()
            .into_iter()
            .map(|x| [x.x, x.y, x.z])
            .collect::<Vec<_>>();
        // let G = Grid::new(
        //     device.clone(),
        //     [res[0], res[1], res[2]],
        //     3,
        //     [0.0, 0.0, 0.0],
        //     h,
        // );
        let G = device.create_buffer::<Mat3>(count).unwrap();
        let G_norm = device.create_buffer::<f32>(count).unwrap();
        let x_bar = device.create_buffer::<Float3>(count).unwrap();
        let build_particle_list_kernel = device
            .create_kernel_async::<(Buffer<Particle>,)>(&|particles: BufferVar<Particle>| {
                let i = dispatch_id().x();
                let p = particles.read(i);
                search.add_to_cell(p.pos(), i);
            })
            .unwrap();
        let diffusion_kernel = device
            .create_kernel_async::<(Buffer<Particle>,)>(&|particles: BufferVar<Particle>| {
                let cell = dispatch_id().xyz();
                let x = phi.pos_i_to_f(cell.int());
                // let phi_v = var!(f32, 0.0);
                let search_cell = search.get_cell(x);
                let avg_x = var!(Float3);
                let sum_w = var!(f32);
                // let avg_r = var!(f32);
                let count = var!(u32);
                search.for_each_particle_in_neighbor(
                    search_cell.uint(),
                    [-1, -1, -1],
                    [1, 1, 1],
                    |j| {
                        let r = x - x_bar.var().read(j);
                        let Gj = G.var().read(j);
                        let Gr = Gj * r;
                        let G_norm = G_norm.var().read(j);
                        // cpu_dbg!(G_norm);
                        let P = |r: Expr<f32>| -> Expr<f32> {
                            if_!(r.cmplt(1.0), {
                                1.0 - 1.5 * r * r + 0.75 * r * r * r
                            }, else {
                                if_!(r.cmplt(2.0), {
                                    let r = 2.0 - r;
                                    0.25 * r * r * r
                                }, else {
                                    0.0.into()
                                })
                            })
                        };
                        // phi_v.store(phi_v.load() + 1.0/h * P(r.length() / h));
                        // phi_v.store(phi_v.load() + G_norm * P(Gr.length()));
                        // let w = G_norm * P(Gr.length());
                        let w = G_norm * w(Gr, 1.0f32.into());
                        if_!(w.cmpgt(0.0),{
                            count.store(count.load() + 1);
                        });

                        sum_w.store(sum_w.load() + w);
                        avg_x.store(avg_x.load() + w * particles.read(j).pos());
                        // avg_r.store(avg_r.load() + w * particles.read(j).radius());
                    },
                );
                if_!(sum_w.load().cmpgt(0.0), {
                    avg_x.store(avg_x.load() / sum_w.load());
                    let avg_x = avg_x.load();
                    phi.set_index(cell, (avg_x - x).length() - radius);
                    // avg_r.store(avg_r.load() / sum_w.load());
                }, else {
                    phi.set_index(cell, h.into());
                });
                // if_!(count.load().cmpgt(0), {
                //     cpu_dbg!(count.load());
                // });
                // let avg_r = avg_r.load();
                // if_!(phi_v.load().cmpgt(0.0), { cpu_dbg!(phi_v.load()) });
            })
            .unwrap();
        let compute_x_bar_kernel = device
            .create_kernel_async::<(Buffer<Particle>,)>(&|particles: BufferVar<Particle>| {
                let lambda = 0.92;
                let i = dispatch_id().x();
                let p = particles.read(i);
                let x = var!(Float3);
                let sum_w = var!(f32);
                search.for_each_neighbor_node(p.pos(), |g| {
                    search.for_each_particle_in_cell(g, |j| {
                        if_!(i.cmpne(j), {
                            let q = particles.read(j);
                            let d = p.pos() - q.pos();
                            let w = w(d, h.into());
                            x.store(x.load() + w * q.pos());
                            sum_w.store(sum_w.load() + w);
                        });
                    })
                });
                if_!(sum_w.load().cmpgt(0.0), {
                    x.store(x.load() / sum_w.load());
                });
                let x = (1.0 - lambda) * p.pos() + lambda * x.load();
                // cpu_dbg!(x);
                x_bar.var().write(i, x);
            })
            .unwrap();
        let compute_G_kernel = device
            .create_kernel_async::<(Buffer<Particle>,)>(&|particles: BufferVar<Particle>| {
                let i = dispatch_id().x();
                let p = particles.read(i);
                let x = var!(Float3);
                let sum_w = var!(f32);
                let node = search.pos_f_to_i(p.pos());
                search.for_each_particle_in_neighbor(node.uint(), [-1, -1, -1], [1, 1, 1], |j| {
                    if_!(i.cmpne(j), {
                        let q = particles.read(j);
                        let d = p.pos() - q.pos();
                        let w = w(d, h.into());
                        x.store(x.load() + w * q.pos());
                        sum_w.store(sum_w.load() + w);
                    });
                });

                if_!(sum_w.load().cmpgt(0.0), {
                    x.store(x.load() / sum_w.load());
                });
                let x = x.load();
                let C = var!(Mat3);
                let N = var!(u32, 0);
                search.for_each_particle_in_neighbor(node.uint(), [-1, -1, -1], [1, 1, 1], |j| {
                    if_!(i.cmpne(j), {
                        let q = particles.read(j);
                        let d = q.pos() - x;
                        let C_ij = d.outer_product(d);
                        let d = p.pos() - q.pos();
                        let w = w(d, h.into());
                        if_!(w.cmpgt(0.0), {
                            N.store(N.load() + 1);
                            //     cpu_dbg!(w);
                            //     cpu_dbg!(C_ij);
                        });
                        C.store(C.load() + C_ij * Mat3Expr::eye(Float3Expr::splat(w)));
                    });
                });
                let C = C.load() * Mat3Expr::eye(Float3Expr::splat(1.0 / sum_w.load()));

                let compute_G = CpuFn::<ComputeGData>::new(|data: &mut ComputeGData| {
                    let C = glam::Mat3::from(data.C).to_cols_array();
                    let mut G = [0.0f32; 9];
                    unsafe {
                        cpp_extra::compute_G(
                            C.as_ptr(),
                            data.N,
                            data.h,
                            G.as_mut_ptr(),
                            &mut data.G_norm,
                        );
                    }
                    data.G = glam::Mat3::from_cols_array(&G).into();
                    // dbg!(data);
                });

                let data = ComputeGDataExpr::new(C, zeroed::<Mat3>(), h, N.load(), 0.0);
                let data = compute_G.call(data);
                let G_i = data.G();
                // if_!(N.load().cmpgt(0), {
                //     cpu_dbg!(N.load());
                //     // cpu_dbg!(G_i);
                // });
                G.var().write(i, G_i);
                G_norm.var().write(i, data.G_norm());
            })
            .unwrap();
        Self {
            res,
            device,
            phi,
            search,
            build_particle_list_kernel,
            compute_x_bar_kernel,
            compute_G_kernel,
            diffusion_kernel,
            x_bar,
            h,
            G,
            GV,
            G_norm,
        }
    }
    fn build_scalar_field(&self, particles: &Buffer<Particle>) -> Vec<f32> {
        self.search.reset_particle_list();
        self.build_particle_list_kernel
            .dispatch([particles.len() as u32, 1, 1], particles)
            .unwrap();
        self.compute_x_bar_kernel
            .dispatch([particles.len() as u32, 1, 1], particles)
            .unwrap();
        self.compute_G_kernel
            .dispatch([particles.len() as u32, 1, 1], particles)
            .unwrap();
        self.diffusion_kernel
            .dispatch(self.phi.res, particles)
            .unwrap();
        self.phi.values.copy_to_vec()
    }
    pub fn save_obj(&self, particles: &Buffer<Particle>, frame: usize, name: &str) {
        let S = self.build_scalar_field(particles);
        // dbg!(&S);
        let path = format!("output_meshes/{}_{}.obj", name, frame);
        let path = CString::new(path).unwrap();
        unsafe {
            let nx = self.phi.res[0];
            let ny = self.phi.res[1];
            let nz = self.phi.res[2];
            dbg!(self.phi.res);
            cpp_extra::save_reconstructed_mesh(
                nx,
                ny,
                nz,
                S.as_ptr(),
                self.GV.as_ptr() as *const f32,
                0.0,
                path.as_ptr(),
            );
        }
    }
}
