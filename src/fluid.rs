use crate::{
    grid::Grid,
    pcgsolver::{PcgSolver, Preconditioner, Stencil},
    *,
};

#[derive(Clone, Copy, Value, Default)]
#[repr(C)]
pub struct Particle {
    pub pos: Float3,
    pub vel: Float3,
    pub mass: f32,
    pub radius: f32,

    pub c_x: Float3,
    pub c_y: Float3,
    pub c_z: Float3,
}
pub mod cell_type {
    pub const FLUID: u32 = 0;
    pub const SOLID: u32 = 1;
}
#[derive(Clone, Copy)]
pub enum ParticleTransfer {
    Pic,
    Flip,
    PicFlip(f32),
    Apic,
}

#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct ControlData {
    pub h: f32,
    // modified for CFL condition
    pub dt: f32,
    pub max_dt: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
}
#[derive(Clone, Copy)]
pub struct SimulationSettings {
    pub dt: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
    pub res: [u32; 3],
    pub h: f32,
    pub dimension: usize,
    pub transfer: ParticleTransfer,
    pub advect: VelocityIntegration,
    pub preconditioner: Preconditioner,
}
pub struct Simulation {
    pub settings: SimulationSettings,
    res_p1: [u32; 3],

    pub u: Grid<f32>,
    pub v: Grid<f32>,
    pub w: Grid<f32>,

    pub tmp_u: Grid<f32>,
    pub tmp_v: Grid<f32>,
    pub tmp_w: Grid<f32>,

    pub liquid_phi: Grid<f32>,

    pub p: Grid<f32>,

    pub div_velocity: Grid<f32>,

    // pub debug_velocity: Grid<Float3>,
    pub dimension: usize,
    pub res: [u32; 3],

    pub solver: Option<PcgSolver>,

    // offset: self, x_p, x_m, y_p, y_m, z_p, z_m
    pub A: Option<Stencil>,
    pub particles_vec: Vec<Particle>,
    pub particles: Option<Buffer<Particle>>,
    advect_particle_kernel: Option<Kernel<(f32,)>>,
    apply_gravity_kernel: Option<Kernel<(f32,)>>,
    build_A_kernel: Option<Kernel<()>>,
    compute_phi_kernel: Option<Kernel<()>>,
    div_velocity_kernel: Option<Kernel<()>>,
    build_particle_list_kernel: Option<Kernel<()>>,
    particle_to_grid_kernel: Option<Kernel<()>>,
    grid_to_particle_kernel: Option<Kernel<()>>,
    velocity_update_kernel: Option<Kernel<(f32,)>>,
    compute_cfl_kernel: Option<Kernel<(Buffer<f32>,)>>,
    zero_kernel: Kernel<(Buffer<f32>,)>,
    pub device: Device,
    pub control: Buffer<ControlData>,
    cfl_tmp: Buffer<f32>,
}
#[derive(Clone, Copy, PartialOrd, PartialEq, Debug)]
pub enum VelocityIntegration {
    Euler,
    RK2,
    RK3,
}

impl Simulation {
    pub fn new(device: Device, settings: SimulationSettings) -> Self {
        let res = settings.res;
        let dimension = settings.dimension;
        let h = settings.h;
        let dt = settings.dt;
        let u_res = [res[0] + 1, res[1], res[2]];
        let v_res = [res[0], res[1] + 1, res[2]];
        let w_res = [res[0], res[1], if dimension == 3 { res[2] + 1 } else { 1 }];
        let u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let tmp_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let tmp_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let tmp_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let make_pressure_grid = || {
            Grid::new(
                device.clone(),
                [res[0], res[1], if dimension == 3 { res[2] } else { 1 }], // boundary is zero
                dimension,
                [0.0, 0.0, 0.0],
                h,
            )
        };
        let liquid_phi = make_pressure_grid();

        let p = make_pressure_grid();
        // let pcg_tmp = make_pressure_grid();
        let div_velocity = make_pressure_grid();
        let control = device
            .create_buffer_from_slice(&[ControlData {
                h,
                max_dt: dt,
                dt,
                max_iterations: settings.max_iterations,
                tolerance: settings.tolerance,
            }])
            .unwrap();
        let zero_kernel = device
            .create_kernel_async::<(Buffer<f32>,)>(&|buf: BufferVar<f32>| {
                let i = dispatch_id().x();
                buf.write(i, 0.0);
            })
            .unwrap();
        Self {
            res_p1: [
                res[0] + 1,
                res[1] + 1,
                if dimension == 3 { res[2] + 1 } else { 1 },
            ],
            settings,
            u,
            v,
            w,
            tmp_u,
            tmp_v,
            tmp_w,
            p,
            liquid_phi,
            solver: None,
            div_velocity,
            dimension,
            res,
            A: None,
            compute_phi_kernel: None,
            build_A_kernel: None,
            particles_vec: Vec::new(),
            particles: None,
            apply_gravity_kernel: None,
            advect_particle_kernel: None,
            div_velocity_kernel: None,
            build_particle_list_kernel: None,
            particle_to_grid_kernel: None,
            grid_to_particle_kernel: None,
            velocity_update_kernel: None,
            compute_cfl_kernel: None,
            cfl_tmp: device.create_buffer(1024).unwrap(),
            device: device.clone(),
            zero_kernel,
            control,
        }
    }
    pub fn h(&self) -> Expr<f32> {
        self.control.var().read(0).h()
    }
    pub fn velocity(&self, p: Expr<Float3>) -> Expr<Float3> {
        if self.dimension == 2 {
            let u = self.u.interpolate(p);
            let v = self.v.interpolate(p);
            make_float3(u, v, 0.0)
        } else {
            let u = self.u.interpolate(p);
            let v = self.v.interpolate(p);
            let w = self.w.interpolate(p);
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
        self.particles = Some(
            self.device
                .create_buffer_from_slice(&self.particles_vec)
                .unwrap(),
        );
        self.u.init_particle_list(self.particles_vec.len());
        self.v.init_particle_list(self.particles_vec.len());
        self.p.init_particle_list(self.particles_vec.len());
        let p_res = self.p.res;
        self.solver = Some(PcgSolver::new(
            self.device.clone(),
            p_res,
            self.settings.max_iterations as usize,
            self.settings.tolerance,
            self.settings.preconditioner,
        ));

        let offsets = [
            Int3::new(0, 0, 0),
            Int3::new(1, 0, 0),
            Int3::new(-1, 0, 0),
            Int3::new(0, 1, 0),
            Int3::new(0, -1, 0),
            Int3::new(0, 0, 1),
            Int3::new(0, 0, -1),
        ];
        if self.dimension == 3 {
            self.w.init_particle_list(self.particles_vec.len());

            self.A = Some(Stencil::new(self.device.clone(), p_res, &offsets));
        } else {
            self.A = Some(Stencil::new(self.device.clone(), p_res, &offsets[..5]));
        }

        self.build_particle_list_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap();
                    let pt = particles.var().read(i);
                    self.u.add_to_cell(pt.pos(), i);
                    self.v.add_to_cell(pt.pos(), i);
                    if self.dimension == 3 {
                        self.w.add_to_cell(pt.pos(), i);
                    }
                    self.p.add_to_cell(pt.pos(), i);
                })
                .unwrap(),
        );
        self.particle_to_grid_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    let cell_idx = dispatch_id().xyz();
                    let particles = self.particles.as_ref().unwrap();
                    let transfer = self.settings.transfer;
                    let (pic_weight, flip_weight, apic_weight) = match transfer {
                        ParticleTransfer::Pic => (1.0, 0.0, 0.0),
                        ParticleTransfer::Flip => (0.0, 1.0, 0.0),
                        ParticleTransfer::PicFlip(flip_weight) => {
                            (1.0 - flip_weight, flip_weight, 0.0)
                        }
                        ParticleTransfer::Apic => (0.0, 0.0, 1.0),
                    };
                    let map = |g: &Grid<f32>, axis: u8| {
                        if_!(!g.oob(cell_idx.int()), {
                            self.transfer_particles_to_grid_impl(
                                g,
                                particles,
                                cell_idx,
                                axis,
                                pic_weight,
                                flip_weight,
                                apic_weight,
                            )
                        });
                    };
                    map(&self.u, 0);
                    map(&self.v, 1);
                    if self.dimension == 3 {
                        map(&self.w, 2);
                    }
                })
                .unwrap(),
        );
        self.grid_to_particle_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    let particles = self.particles.as_ref().unwrap();
                    let pt_idx = dispatch_id().x();
                    let transfer = self.settings.transfer;
                    let (pic_weight, flip_weight, apic_weight) = match transfer {
                        ParticleTransfer::Pic => (1.0, 0.0, 0.0),
                        ParticleTransfer::Flip => (0.0, 1.0, 0.0),
                        ParticleTransfer::PicFlip(flip_weight) => {
                            (1.0 - flip_weight, flip_weight, 0.0)
                        }
                        ParticleTransfer::Apic => (0.0, 0.0, 1.0),
                    };
                    let map = |g: &Grid<f32>, old_g: &Grid<f32>, axis: u8| {
                        self.transfer_grid_to_particles_impl(
                            g,
                            old_g,
                            particles,
                            pt_idx,
                            axis,
                            pic_weight,
                            flip_weight,
                            apic_weight,
                        )
                    };
                    map(&self.u, &self.tmp_u, 0);
                    map(&self.v, &self.tmp_v, 1);
                    if self.dimension == 3 {
                        map(&self.w, &self.tmp_w, 2);
                    }
                })
                .unwrap(),
        );
        self.compute_phi_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    let node = dispatch_id();
                    let particles = self.particles.as_ref().unwrap();
                    let phi = var!(f32, self.h() * 1e3);
                    let node_pos = self.p.pos_i_to_f(node.int());
                    self.p.for_each_particle_in_neighbor(node, |pt_idx| {
                        let pt = particles.var().read(pt_idx);
                        let new_phi = pt.pos().distance(node_pos) - pt.radius();
                        phi.store(phi.load().min(new_phi));
                    });
                    self.liquid_phi.set_index(node, phi.load());
                })
                .unwrap(),
        );
        self.advect_particle_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| {
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap();
                    let pt = particles.var().read(i);
                    pt.set_pos(self.integrate_velocity(pt.pos(), dt, VelocityIntegration::RK2));
                    particles.var().write(i, pt);
                })
                .unwrap(),
        );
        self.apply_gravity_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| {
                    let p = dispatch_id();
                    let v = self.v.at_index(p);
                    self.v.set_index(p, v - 0.981 * dt);
                })
                .unwrap(),
        );
        self.build_A_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.build_A_impl();
                })
                .unwrap(),
        );

        self.div_velocity_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.divergence(&self.u, &self.v, &self.w, &self.div_velocity)
                })
                .unwrap(),
        );
        self.velocity_update_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| self.update_velocity_impl(dt))
                .unwrap(),
        );
        self.compute_cfl_kernel = Some(
            self.device
                .create_kernel_async::<(Buffer<f32>,)>(&|tmp: BufferVar<f32>| {
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap().var();
                    let pt = particles.read(i);
                    let v = pt.vel().length();
                    tmp.atomic_fetch_max(i % tmp.len(), v);
                })
                .unwrap(),
        );
    }
    pub fn transfer_particles_to_grid_impl(
        &self,
        g: &Grid<f32>,
        particles: &Buffer<Particle>,
        cell_idx: Expr<Uint3>,
        axis: u8,
        pic_weight: f32,
        flip_weight: f32,
        apic_weight: f32,
    ) {
        let weight_sum = pic_weight + flip_weight + apic_weight;
        let pic_weight = pic_weight / weight_sum;
        let flip_weight = flip_weight / weight_sum;
        let apic_weight = apic_weight / weight_sum;

        let cell_pos = g.pos_i_to_f(cell_idx.int());
        let pic_flip_mv = var!(f32); // momentum
        let apic_flip_mv = var!(f32); // momentum

        let m = var!(f32); // mass

        g.for_each_particle_in_neighbor(cell_idx, |pt_idx| {
            let pt = particles.var().read(pt_idx);
            let m_p = pt.mass();
            let v_p = pt.vel();
            let offset = (pt.pos() - cell_pos) / g.dx;
            let w_p = trilinear_weight(offset, self.dimension);

            let v_pa = match axis {
                0 => v_p.x(),
                1 => v_p.y(),
                2 => v_p.z(),
                _ => unreachable!(),
            };
            let c_pa = match axis {
                0 => pt.c_x(),
                1 => pt.c_y(),
                2 => pt.c_z(),
                _ => unreachable!(),
            };

            pic_flip_mv.store(pic_flip_mv.load() + v_pa * m_p * w_p);

            apic_flip_mv
                .store(apic_flip_mv.load() + m_p * w_p * (v_pa + c_pa.dot(cell_pos - pt.pos())));
            m.store(m.load() + m_p * w_p);
        });

        let pic_flip_v = pic_flip_mv.load() / m.load();
        let apic_flip_v = apic_flip_mv.load() / m.load();
        // TODO: check this
        let final_v = (pic_weight + flip_weight) * pic_flip_v + apic_weight * apic_flip_v;
        g.set_index(cell_idx, final_v);
    }
    fn transfer_grid_to_particles_impl(
        &self,
        g: &Grid<f32>,
        old_g: &Grid<f32>,
        particles: &Buffer<Particle>,
        pt_idx: Expr<u32>,
        axis: u8,
        pic_weight: f32,
        flip_weight: f32,
        apic_weight: f32,
    ) {
        let particles = particles.var();
        let pt = particles.read(pt_idx);
        let weight_sum = pic_weight + flip_weight + apic_weight;
        let pic_weight = pic_weight / weight_sum;
        let flip_weight = flip_weight / weight_sum;
        let apic_weight = apic_weight / weight_sum;

        let v_p = pt.vel();

        let pic_v_pa = var!(f32);
        let flip_v_pa = var!(f32);

        let apic_c_pa = var!(Float3);

        g.for_each_neighbor_node(pt.pos(), |node| {
            let node_pos = g.pos_i_to_f(node.int());
            let v_i = g.at_index(node);
            let old_v_i = old_g.at_index(node);
            let diff_v = v_i - old_v_i;
            let offset = (pt.pos() - node_pos) / g.dx;
            let w_pa = trilinear_weight(offset, self.dimension);
            // TODO: should i divide dx?
            let grad_w_pa = grad_trilinear_weight(offset, self.dimension) / g.dx;

            flip_v_pa.store(flip_v_pa.load() + diff_v * w_pa);
            pic_v_pa.store(pic_v_pa.load() + v_i * w_pa);
            apic_c_pa.store(apic_c_pa.load() + grad_w_pa * v_i);
        });
        // TODO: check this
        let v_pa = (apic_weight + pic_weight) * pic_v_pa.load() + flip_weight * flip_v_pa.load();
        let pt = var!(Particle, pt);
        match axis {
            0 => {
                pt.set_vel(v_p.set_x(v_pa));
                pt.set_c_x(apic_c_pa);
            }
            1 => {
                pt.set_vel(v_p.set_y(v_pa));
                pt.set_c_y(apic_c_pa);
            }
            2 => {
                pt.set_vel(v_p.set_z(v_pa));
                pt.set_c_z(apic_c_pa);
            }
            _ => unreachable!(),
        }
        particles.write(pt_idx, pt.load());
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
        // cpu_dbg!(x);
        // cpu_dbg!(d);
        div.set_index(x, d);
    }

    pub fn build_A_impl(&self) {
        let x = dispatch_id();
        let i = self.p.linear_index(x);
        let A = self.A.as_ref().unwrap();
        let A_coeff = A.coeff.var();
        let h = self.h();
        let h2 = h * h;
        if self.dimension == 2 {
            assert_eq!(A.offsets.len(), 5);
            A_coeff.write(i * A.offsets.len() as u32, 4.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 1, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 2, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 3, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 4, -1.0 / h2);
        } else {
            assert_eq!(A.offsets.len(), 7);
            A_coeff.write(i * A.offsets.len() as u32, 6.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 1, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 2, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 3, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 4, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 5, -1.0 / h2);
            A_coeff.write(i * A.offsets.len() as u32 + 6, -1.0 / h2);
        }
    }
    pub fn apply_A(&self, P: &Grid<f32>, div2: &Grid<f32>, sub: bool) {
        assert_eq!(P.res, div2.res);
        set_block_size([64, 64, 64]);
        let x = dispatch_id();
        let d = if self.dimension == 2 {
            let p = P.at_index(x);
            let x = x.int();
            let p_x = P.at_index_or_zero(x + make_int3(1, 0, 0));
            let p_y = P.at_index_or_zero(x + make_int3(0, 1, 0));
            let p_xm = P.at_index_or_zero(x + make_int3(-1, 0, 0));
            let p_ym = P.at_index_or_zero(x + make_int3(0, -1, 0));
            let d = (-(p_x + p_y + p_xm + p_ym) + 4.0 * p) / (self.h() * self.h());
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

    pub fn advect_particle(&self, dt: f32) {
        self.advect_particle_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1], &dt)
            .unwrap();
    }
    pub fn apply_ext_forces(&self, dt: f32) {
        self.apply_gravity_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.v.res, &dt)
            .unwrap();
    }
    pub fn compute_div_velocity(&self) {
        self.div_velocity_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.div_velocity.res)
            .unwrap();
    }
    pub fn solve_pressure(&self, dt: f32) {
        self.compute_div_velocity();

        let solver = self.solver.as_ref().unwrap();
        let A = self.A.as_ref().unwrap();
        solver
            .zero
            .dispatch([A.coeff.len() as u32, 1, 1], &A.coeff)
            .unwrap();
        self.build_A_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
        let i = solver.solve(
            self.A.as_ref().unwrap(),
            &self.div_velocity.values,
            &self.p.values,
        );
        if i.is_none() {
            panic!("pressure solver failed");
        }
        self.velocity_update_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1, &dt)
            .unwrap();
    }
    fn transfer_particles_to_grid(&self) {
        self.build_particle_list_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();

        self.particle_to_grid_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1)
            .unwrap();
        let stream = self.device.default_stream();
        stream.with_scope(|s| {
            s.submit([
                self.u.values.copy_to_buffer_async(&self.tmp_u.values),
                self.v.values.copy_to_buffer_async(&self.tmp_v.values),
                self.w.values.copy_to_buffer_async(&self.tmp_w.values),
            ])
            .unwrap();
        });
    }
    fn transfer_grid_to_particles(&self) {
        self.grid_to_particle_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();
    }
    pub fn step(&self) {
        let settings = self.control.copy_to_vec()[0];
        let dt = settings.dt;
        self.advect_particle(dt);
        self.transfer_particles_to_grid();
        self.apply_ext_forces(dt);

        self.solve_pressure(dt);
        self.transfer_grid_to_particles();
    }
    // solve for (1 - alpha) * cur_phi + alpha * neighbor_phi = 0
    //  cur_phi - alpha * cur_phi + alpha * neighbor_phi = 0
    //  -alpha * (cur_phi - neighbor_phi) = -cur_phi
    //  alpha = -cur_phi / (cur_phi - neighbor_phi)
    fn free_surface_alpha(&self, cur_phi: Expr<f32>, neighbor_phi: Expr<f32>) -> Expr<f32> {
        cur_phi / (neighbor_phi - cur_phi)
    }
    fn update_velocity_impl(&self, dt: Expr<f32>) {
        // u = u - dt * grad(p)
        let i = dispatch_id();

        let update = |u: &Grid<f32>, axis: u8| {
            let i = i.int();
            if_!(!u.oob(i), {
                let off = match axis {
                    0 => make_int3(1, 0, 0),
                    1 => make_int3(0, 1, 0),
                    2 => make_int3(0, 0, 1),
                    _ => unreachable!(),
                };
                let i_a = i.at(axis as usize);
                if_!(i_a.cmpgt(0) & i_a.cmplt(u.res[axis as usize] - 1), {
                    let grad_pressure =
                        (self.p.at_index(i.uint()) - self.p.at_index((i - off).uint())) / self.h();
                    let u_cur = u.at_index(i.uint());
                    u.set_index(i.uint(), u_cur - dt * grad_pressure);
                }, else{
                    // set component to zero
                    u.set_index(i.uint(), 0.0.into());
                });
            });
        };
        update(&self.u, 0);
        update(&self.v, 1);
        if self.dimension == 3 {
            update(&self.w, 2);
        }
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
