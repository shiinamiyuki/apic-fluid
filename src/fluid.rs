use crate::{
    grid::Grid,
    sparse::{solver::PcgSolver, SparseMatrix},
    *,
};
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct Particle {
    pub pos: Float3,
    pub vel: Float3,
    pub mass: f32,
    pub radius: f32,
}
pub mod cell_type {
    pub const FLUID: u32 = 0;
    pub const SOLID: u32 = 1;
}
#[derive(Clone, Copy)]
pub enum ParticleTransfer {
    None,
    Pic,
    Flip,
    PicFlip(f32),
    Apic,
}

#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct SimulationSettings {
    pub h: f32,
    pub dt: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
}
pub struct Simulation {
    pub u: Grid<f32>,
    pub v: Grid<f32>,
    pub w: Grid<f32>,

    pub p: Grid<f32>,

    pub pcg_r: Grid<f32>,
    pub pcg_s: Grid<f32>,
    pub pcg_z: Grid<f32>,
    // pub pcg_tmp: Grid<f32>,
    pub div_velocity: Grid<f32>,

    // pub debug_velocity: Grid<Float3>,
    pub dimension: usize,
    pub res: [u32; 3],
    pub solver: PcgSolver,
    pub A: Option<SparseMatrix>,
    pub particles_vec: Vec<Particle>,
    pub particles: Option<Buffer<Particle>>,
    pub advect_particle_kernel: Option<Kernel<(Buffer<f32>,)>>,
    pub apply_gravity_kernel: Option<Kernel<(Buffer<f32>,)>>,
    pub pcg_Ax_kernel: Option<Kernel<()>>,
    pub pcg_As_kernel: Option<Kernel<()>>,
    pub incomplete_poission_kernel: Option<Kernel<()>>,
    pub div_velocity_kernel: Option<Kernel<()>>,
    pub add_scaled_kernel: Option<Kernel<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>>,
    pub dot_kernel: Option<Kernel<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>>,
    pub abs_max_kernel: Option<Kernel<(Buffer<f32>, Buffer<f32>)>>,
    pub build_particle_list_kernel: Option<Kernel<()>>,
    pub particle_to_grid_kernel: Option<Kernel<()>>,
    pub grid_to_particle_kernel: Option<Kernel<()>>,
    pub device: Device,
    pub settings: Buffer<SimulationSettings>,
}
#[derive(Clone, Copy, PartialOrd, PartialEq, Debug)]
pub enum VelocityIntegration {
    Euler,
    RK2,
    RK3,
}
#[derive(Clone, Copy, PartialOrd, PartialEq, Debug)]
pub enum Preconditioner {
    Identity,
    DiagJacobi,
    IncompletePoisson,
}
impl Simulation {
    pub fn new(device: Device, res: [u32; 3], h: f32, dimension: usize) -> Self {
        let u = Grid::new(device.clone(), res, dimension, [0.5 * h, 0.0, 0.0]);
        let v = Grid::new(device.clone(), res, dimension, [0.0, 0.5 * h, 0.0]);
        let w = Grid::new(device.clone(), res, dimension, [0.0, 0.0, 0.5 * h]);

        let make_pressure_grid = || {
            Grid::new(
                device.clone(),
                [res[0] - 1, res[1] - 1, res[2] - 1], // boundary is zero
                dimension,
                [0.0, 0.0, 0.0],
            )
        };
        let p = make_pressure_grid();
        let pcg_s = make_pressure_grid();
        let pcg_r = make_pressure_grid();
        let pcg_z = make_pressure_grid();
        // let pcg_tmp = make_pressure_grid();
        let div_velocity = make_pressure_grid();
        let settings = device
            .create_buffer_from_slice(&[SimulationSettings {
                h,
                dt: 0.01,
                max_iterations: 100,
                tolerance: 1e-5,
            }])
            .unwrap();
        Self {
            u,
            v,
            w,
            p,
            pcg_s,
            pcg_r,
            pcg_z,
            // pcg_tmp,
            div_velocity,
            dimension,
            res,
            solver: PcgSolver::new(device.clone(), (res[0] * res[1] * res[2]) as usize),
            A: None,
            particles_vec: Vec::new(),
            particles: None,
            advect_particle_kernel: None,
            apply_gravity_kernel: None,
            pcg_Ax_kernel: None,
            pcg_As_kernel: None,
            incomplete_poission_kernel: None,
            div_velocity_kernel: None,
            dot_kernel: None,
            abs_max_kernel: None,
            add_scaled_kernel: None,
            build_particle_list_kernel: None,
            particle_to_grid_kernel: None,
            grid_to_particle_kernel: None,
            device,
            settings,
        }
    }
    pub fn h(&self) -> Expr<f32> {
        self.settings.var().read(0).h()
    }
    pub fn velocity(&self, p: Expr<Float3>) -> Expr<Float3> {
        if self.dimension == 2 {
            let u = self.u.bilinear(p / self.h());
            let v = self.v.bilinear(p / self.h());
            make_float3(u, v, 0.0)
        } else {
            let u = self.u.bilinear(p / self.h());
            let v = self.v.bilinear(p / self.h());
            let w = self.w.bilinear(p / self.h());
            make_float3(u, v, w)
        }
    }
    pub fn integrate_velocity(
        &self,
        p: Expr<Float3>,
        dt: Expr<f32>,
        scheme: VelocityIntegration,
    ) -> Expr<Float3> {
        match scheme {
            VelocityIntegration::Euler => p + self.velocity(p) * dt,
            VelocityIntegration::RK2 => {
                let k1 = dt * self.velocity(p);
                let k2 = dt * self.velocity(p + k1 * 0.5);
                p + k2
            }
            VelocityIntegration::RK3 => {
                let k1 = dt * self.velocity(p);
                let k2 = dt * self.velocity(p + k1 * 0.5);
                let k3 = dt * self.velocity(p - k1 + k2 * 2.0);
                p + (k1 + k2 * 4.0 + k3) / 6.0
            }
        }
    }
    pub fn commit(&mut self) {
        self.build_particle_list_kernel = Some(
            self.device
                .create_kernel::<()>(&|| {
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap();
                    let pt = particles.var().read(i);
                    self.u.add_to_cell(pt.pos(), i);
                    self.v.add_to_cell(pt.pos(), i);
                    if self.dimension == 3 {
                        self.w.add_to_cell(pt.pos(), i);
                    }
                })
                .unwrap(),
        );
        self.particle_to_grid_kernel = Some(
            self.device
                .create_kernel::<()>(&|| {
                    let cell = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap();
                    todo!()
                })
                .unwrap(),
        );
        self.advect_particle_kernel = Some(
            self.device
                .create_kernel::<(Buffer<f32>,)>(&|dt: BufferVar<f32>| {
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap();
                    let pt = particles.var().read(i);
                    let dt = dt.read(0);
                    pt.set_pos(self.integrate_velocity(pt.pos(), dt, VelocityIntegration::RK2));
                    particles.var().write(i, pt);
                })
                .unwrap(),
        );
        self.apply_gravity_kernel = Some(
            self.device
                .create_kernel::<(Buffer<f32>,)>(&|dt: BufferVar<f32>| {
                    let p = dispatch_id();
                    let v = self.v.at_index(p);
                    self.v.set_index(p, v - 0.981 * dt.read(0));
                })
                .unwrap(),
        );
        self.pcg_Ax_kernel = Some(
            self.device
                .create_kernel::<()>(&|| self.apply_A(&self.p, &self.pcg_r, true))
                .unwrap(),
        );
        self.pcg_As_kernel = Some(
            self.device
                .create_kernel::<()>(&|| self.apply_A(&self.pcg_s, &self.pcg_z, false))
                .unwrap(),
        );
        self.incomplete_poission_kernel = Some(
            self.device
                .create_kernel::<()>(&|| self.incomplete_poisson(&self.pcg_r, &self.pcg_z))
                .unwrap(),
        );
        self.dot_kernel =
            Some(
                self.device
                    .create_kernel::<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                        &|a: BufferVar<f32>, b: BufferVar<f32>, result: BufferVar<f32>| {
                            let i = dispatch_id().x();
                            let block_size = 256;
                            result.atomic_fetch_add(i % block_size, a.read(i) * b.read(i));
                        },
                    )
                    .unwrap(),
            );
        self.abs_max_kernel = Some(
            self.device
                .create_kernel::<(Buffer<f32>, Buffer<f32>)>(
                    &|a: BufferVar<f32>, result: BufferVar<f32>| {
                        let i = dispatch_id().x();
                        let block_size = 256;
                        let cur = a.read(i).abs();
                        let j = i % block_size;
                        result.atomic_fetch_max(j, cur);
                    },
                )
                .unwrap(),
        );
        self.add_scaled_kernel = Some(
            self.device
                .create_kernel::<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                    &|k: BufferVar<f32>, a: BufferVar<f32>, out: BufferVar<f32>| {
                        let i = dispatch_id().x();
                        out.write(i, a.read(i) + k.read(0) * a.read(i));
                    },
                )
                .unwrap(),
        );
        self.div_velocity_kernel = Some(
            self.device
                .create_kernel::<()>(&|| {
                    self.divergence(&self.u, &self.v, &self.w, &self.div_velocity)
                })
                .unwrap(),
        );
    }
    pub fn divergence(&self, u: &Grid<f32>, v: &Grid<f32>, w: &Grid<f32>, div: &Grid<f32>) {
        let x = dispatch_id();
        let offset = make_uint2(0, 1);
        let d: Expr<f32> = if self.dimension == 2 {
            let du = u.at_index(x + offset.yxx()) - u.at_index(x);
            let dv = v.at_index(x + offset.xyx()) - v.at_index(x);
            (du + dv) / self.h()
        } else {
            let du = u.at_index(x + offset.yxx()) - u.at_index(x);
            let dv = v.at_index(x + offset.xyx()) - v.at_index(x);
            let dw = w.at_index(x + offset.xxy()) - w.at_index(x);
            (du + dv + dw) / self.h()
        };
        div.set_index(x, d);
    }
    pub fn incomplete_poisson(&self, P: &Grid<f32>, z: &Grid<f32>) {
        let x = dispatch_id();
        let d = if self.dimension == 2 {
            let p = P.at_index(x);
            let x = x.int();

            let p_x = P.at_index_or_zero(x + make_int3(1, 0, 0));
            let p_y = P.at_index_or_zero(x + make_int3(0, 1, 0));
            let p_xm = P.at_index_or_zero(x + make_int3(-1, 0, 0));
            let p_ym = P.at_index_or_zero(x + make_int3(0, -1, 0));
            let d = ((p_x + p_y + p_xm + p_ym) * (1.0 / 4.0f32) + (9.0 / 8.0f32) * p)
                / (self.h() * self.h());
            d
        } else {
            todo!()
        };
        z.set_index(x.uint(), d);
    }
    pub fn apply_A(&self, P: &Grid<f32>, div2: &Grid<f32>, sub: bool) {
        let x = dispatch_id();
        let d = if self.dimension == 2 {
            let p = P.at_index(x);
            let x = x.int();
            let p_x = P.at_index_or_zero(x + make_int3(1, 0, 0));
            let p_y = P.at_index_or_zero(x + make_int3(0, 1, 0));
            let p_xm = P.at_index_or_zero(x + make_int3(-1, 0, 0));
            let p_ym = P.at_index_or_zero(x + make_int3(0, -1, 0));
            let d = (p_x + p_y + p_xm + p_ym - 4.0 * p) / (self.h() * self.h());
            d
        } else {
            todo!()
        };
        if !sub {
            div2.set_index(x.uint(), d);
        } else {
            div2.set_index(x.uint(), div2.at_index(x.uint()) - d);
        }
    }
    pub fn r_eq_r_sub_A_x(&self) {
        self.pcg_Ax_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
    }
    pub fn z_eq_A_s(&self) {
        self.pcg_As_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
    }
    pub fn dot(&self, a: &Grid<f32>, b: &Grid<f32>) -> f32 {
        let tmp = self
            .device
            .create_buffer_from_fn::<f32>(256, |_| 0.0)
            .unwrap();
        self.dot_kernel
            .as_ref()
            .unwrap()
            .dispatch(
                [a.res[0] * a.res[1] * a.res[2], 1, 1],
                &a.values,
                &b.values,
                &tmp,
            )
            .unwrap();
        tmp.copy_to_vec().iter().sum()
    }
    pub fn abs_max(&self, a: &Grid<f32>) -> f32 {
        let tmp = self
            .device
            .create_buffer_from_fn::<f32>(256, |_| 0.0)
            .unwrap();
        self.abs_max_kernel
            .as_ref()
            .unwrap()
            .dispatch([a.res[0] * a.res[1] * a.res[2], 1, 1], &a.values, &tmp)
            .unwrap();
        *tmp.copy_to_vec()
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
    pub fn add_scaled(&self, k: f32, a: &Grid<f32>, out: &Grid<f32>) {
        let k = self.device.create_buffer_from_slice(&[k]).unwrap();
        self.add_scaled_kernel
            .as_ref()
            .unwrap()
            .dispatch(
                [a.res[0] * a.res[1] * a.res[2], 1, 1],
                &k,
                &a.values,
                &out.values,
            )
            .unwrap();
    }
    // solve Px = b
    pub fn apply_preconditioner(&self) {
        let use_ipp = true;
        if !use_ipp {
            // first try identity
            self.pcg_r.values.copy_to_buffer(&self.pcg_z.values);
        } else {
            // then try incomplete poisson
            self.incomplete_poission_kernel
                .as_ref()
                .unwrap()
                .dispatch(self.pcg_r.res)
                .unwrap();
        }
    }
    pub fn advect_particle(&self, dt: &Buffer<f32>) {
        self.advect_particle_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1], dt)
            .unwrap();
    }
    pub fn apply_ext_forces(&self, dt: &Buffer<f32>) {
        self.apply_gravity_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.v.res, dt)
            .unwrap();
    }
    pub fn compute_div_velocity(&self) {
        self.div_velocity_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.div_velocity.res)
            .unwrap();
    }
    pub fn solve_pressure(&self) {
        self.compute_div_velocity();
        let iters = self.pressure_pcg_solver();
        if iters.is_none() {
            panic!("pressure solver failed");
        }
    }
    pub fn pressure_pcg_solver(&self) -> Option<u32> {
        let settings = self.settings.copy_to_vec()[0];
        self.div_velocity.values.copy_to_buffer(&self.pcg_r.values);
        self.r_eq_r_sub_A_x();
        let residual = self.abs_max(&self.pcg_r);
        if residual < settings.tolerance {
            return Some(0);
        }
        let mut rho = self.dot(&self.pcg_r, &self.pcg_z);
        if rho == 0.0 || rho.is_nan() {
            return None;
        }
        self.apply_preconditioner();
        self.pcg_z.values.copy_to_buffer(&self.pcg_s.values);
        for i in 0..settings.max_iterations {
            self.z_eq_A_s();
            let alpha = rho / self.dot(&self.pcg_s, &self.pcg_z);
            self.add_scaled(alpha, &self.pcg_s, &self.p);
            self.add_scaled(-alpha, &self.pcg_z, &self.pcg_r);
            let residual = self.abs_max(&self.pcg_r);
            if residual < settings.tolerance {
                return Some(i + 1);
            }
            self.apply_preconditioner();
            let rho_new = self.dot(&self.pcg_r, &self.pcg_z);
            let beta = rho_new / rho;
            self.add_scaled(beta, &self.pcg_s, &self.pcg_z);
            rho = rho_new;
            self.pcg_z.values.copy_to_buffer(&self.pcg_s.values);
        }
        None
    }

    fn transfer_particles_to_grid(&self) {
        self.build_particle_list_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();
    }
    fn transfter_grid_to_particles(&self) {}
    pub fn step(&self) {
        let settings = self.settings.copy_to_vec()[0];
        let dt = self
            .device
            .create_buffer_from_slice(&[settings.dt])
            .unwrap();
        self.advect_particle(&dt);
        self.transfer_particles_to_grid();
        self.apply_ext_forces(&dt);

        self.solve_pressure();
        self.transfter_grid_to_particles();
    }

    pub fn dump_velocity_field_2d(&self, image_path: &str) {
        assert_eq!(self.dimension, 2);
        use exr::prelude::*;
        let v = self.v.values.copy_to_vec();
        let u = self.u.values.copy_to_vec();
        write_rgb_file(
            image_path,
            self.res[0] as usize,
            self.res[1] as usize,
            |x, y| {
                let i = (y * self.res[0] as usize + x) as usize;
                let v = v[i];
                let u = u[i];
                let mag = (v * v + u * u).sqrt();
                (mag, mag, mag)
            },
        )
        .unwrap();
    }
}
