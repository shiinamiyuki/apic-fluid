use crate::{grid::Grid, pcgsolver::Stencil, *};
const PAR_REDUCE_BLOCK_SIZE: usize = 1024;
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
    pub u: Grid<f32>,
    pub v: Grid<f32>,
    pub w: Grid<f32>,

    pub tmp_u: Grid<f32>,
    pub tmp_v: Grid<f32>,
    pub tmp_w: Grid<f32>,

    pub liquid_phi: Grid<f32>,

    pub p: Grid<f32>,

    pub pcg_r: Grid<f32>,
    pub pcg_s: Grid<f32>,
    pub pcg_z: Grid<f32>,
    // pub pcg_tmp: Grid<f32>,
    pub div_velocity: Grid<f32>,

    // pub debug_velocity: Grid<Float3>,
    pub dimension: usize,
    pub res: [u32; 3],

    // offset: self, x_p, x_m, y_p, y_m, z_p, z_m
    pub A: Option<Stencil>,
    pub particles_vec: Vec<Particle>,
    pub particles: Option<Buffer<Particle>>,
    pub advect_particle_kernel: Option<Kernel<(Buffer<f32>,)>>,
    pub apply_gravity_kernel: Option<Kernel<(Buffer<f32>,)>>,
    pub pcg_Ax_kernel: Option<Kernel<()>>,
    pub pcg_As_kernel: Option<Kernel<()>>,
    pub incomplete_poission_kernel: Option<Kernel<()>>,
    pub build_A_kernel: Option<Kernel<()>>,
    pub div_velocity_kernel: Option<Kernel<()>>,
    pub add_scaled_kernel: Option<Kernel<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>>,
    pub dot_kernel: Option<Kernel<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>>,
    pub abs_max_kernel: Option<Kernel<(Buffer<f32>, Buffer<f32>)>>,
    pub build_particle_list_kernel: Option<Kernel<()>>,
    pub particle_to_grid_kernel: Option<Kernel<()>>,
    pub grid_to_particle_kernel: Option<Kernel<()>>,
    pub device: Device,
    pub control: Buffer<ControlData>,
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
    IncompleteCholesky,
}
impl Simulation {
    pub fn new(device: Device, settings: SimulationSettings) -> Self {
        let res = settings.res;
        let dimension = settings.dimension;
        let h = settings.h;
        let dt = settings.dt;
        let u = Grid::new(device.clone(), res, dimension, [0.5 * h, 0.0, 0.0], h);
        let v = Grid::new(device.clone(), res, dimension, [0.0, 0.5 * h, 0.0], h);
        let w = Grid::new(device.clone(), res, dimension, [0.0, 0.0, 0.5 * h], h);

        let tmp_u = Grid::new(device.clone(), res, dimension, [0.5 * h, 0.0, 0.0], h);
        let tmp_v = Grid::new(device.clone(), res, dimension, [0.0, 0.5 * h, 0.0], h);
        let tmp_w = Grid::new(device.clone(), res, dimension, [0.0, 0.0, 0.5 * h], h);

        let liquid_phi = Grid::new(device.clone(), res, dimension, [0.0, 0.0, 0.0], h);

        let make_pressure_grid = || {
            Grid::new(
                device.clone(),
                [
                    res[0] - 1,
                    res[1] - 1,
                    if dimension == 3 { res[2] - 1 } else { 1 },
                ], // boundary is zero
                dimension,
                [0.0, 0.0, 0.0],
                h,
            )
        };
        let p = make_pressure_grid();
        let pcg_s = make_pressure_grid();
        let pcg_r = make_pressure_grid();
        let pcg_z = make_pressure_grid();
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
        Self {
            settings,
            u,
            v,
            w,
            tmp_u,
            tmp_v,
            tmp_w,
            p,
            liquid_phi,
            pcg_s,
            pcg_r,
            pcg_z,
            // pcg_tmp,
            div_velocity,
            dimension,
            res,
            A: None,
            build_A_kernel: None,
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
            control,
        }
    }
    pub fn h(&self) -> Expr<f32> {
        self.control.var().read(0).h()
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
        self.u.init_particle_list(self.particles_vec.len());
        self.v.init_particle_list(self.particles_vec.len());
        let p_res = self.p.res;
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
        self.particles = Some(
            self.device
                .create_buffer_from_slice(&self.particles_vec)
                .unwrap(),
        );
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
                        self.transfer_particles_to_grid_impl(
                            g,
                            particles,
                            cell_idx,
                            axis,
                            pic_weight,
                            flip_weight,
                            apic_weight,
                        )
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
        self.advect_particle_kernel = Some(
            self.device
                .create_kernel_async::<(Buffer<f32>,)>(&|dt: BufferVar<f32>| {
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
                .create_kernel_async::<(Buffer<f32>,)>(&|dt: BufferVar<f32>| {
                    let p = dispatch_id();
                    let v = self.v.at_index(p);
                    self.v.set_index(p, v - 0.981 * dt.read(0));
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
        self.pcg_Ax_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| self.apply_A(&self.p, &self.pcg_r, true))
                .unwrap(),
        );
        self.pcg_As_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| self.apply_A(&self.pcg_s, &self.pcg_z, false))
                .unwrap(),
        );
        self.incomplete_poission_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| self.incomplete_poisson(&self.pcg_r, &self.pcg_z))
                .unwrap(),
        );
        self.dot_kernel = Some(
            self.device
                .create_kernel_async::<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                    &|a: BufferVar<f32>, b: BufferVar<f32>, result: BufferVar<f32>| {
                        set_block_size([PAR_REDUCE_BLOCK_SIZE as u32 + 1, 1, 1]);
                        let i = dispatch_id().x();
                        let block_size = PAR_REDUCE_BLOCK_SIZE as u32;
                        result.atomic_fetch_add(i % block_size, a.read(i) * b.read(i));
                    },
                )
                .unwrap(),
        );
        self.abs_max_kernel = Some(
            self.device
                .create_kernel_async::<(Buffer<f32>, Buffer<f32>)>(
                    &|a: BufferVar<f32>, result: BufferVar<f32>| {
                        set_block_size([PAR_REDUCE_BLOCK_SIZE as u32 + 1, 1, 1]);
                        let i = dispatch_id().x();
                        let block_size = PAR_REDUCE_BLOCK_SIZE as u32;
                        let cur = a.read(i).abs();
                        let j = i % block_size;
                        result.atomic_fetch_max(j, cur);
                    },
                )
                .unwrap(),
        );
        // FIX!
        self.add_scaled_kernel = Some(
            self.device
                .create_kernel_async::<(Buffer<f32>, Buffer<f32>, Buffer<f32>)>(
                    &|k: BufferVar<f32>, a: BufferVar<f32>, out: BufferVar<f32>| {
                        set_block_size([512, 1, 1]);
                        let i = dispatch_id().x();
                        out.write(i, out.read(i) + k.read(0) * a.read(i));
                    },
                )
                .unwrap(),
        );
        self.div_velocity_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.divergence(&self.u, &self.v, &self.w, &self.div_velocity)
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
    pub fn incomplete_poisson(&self, P: &Grid<f32>, z: &Grid<f32>) {
        // TODO: check this
        /*
        Incomplete Poisson:
        If Ax=b can be written as stencil operation:
            b_{i,j} = m_{i-1, j} * x_{i-1,j} + m_{i, j-1} * x_{i,j-1} + m_{i+1, j} * x_{i+1,j}+ m_{i, j+1} * x_{i,j+1} + m_{i,j} * x_{i,j}
        Then to apply the preconditioner, simply compute:
            z_{i,j} = todo!()
        */
        set_block_size([64, 64, 64]);
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
    pub fn build_A_impl(&self) {
        let x = dispatch_id();
        let i = self.p.linear_index(x);
        let A = self.A.as_ref().unwrap();
        let A_coeff = A.coeff.var();

        if self.dimension == 2 {
            assert_eq!(A.offsets.len(), 5);
            A_coeff.write(i * A.offsets.len() as u32, 4.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 1, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 2, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 3, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 4, -1.0 / self.h());
        } else {
            assert_eq!(A.offsets.len(), 7);
            A_coeff.write(i * A.offsets.len() as u32, 6.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 1, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 2, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 3, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 4, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 5, -1.0 / self.h());
            A_coeff.write(i * A.offsets.len() as u32 + 6, -1.0 / self.h());
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
            .create_buffer_from_fn::<f32>(PAR_REDUCE_BLOCK_SIZE, |_| 0.0)
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
        // let a = a.values.copy_to_vec();
        // let b = b.values.copy_to_vec();
        // a.par_iter().zip(b.par_iter()).map(|(a, b)| a * b).sum()
    }
    pub fn abs_max(&self, a: &Grid<f32>) -> f32 {
        let tmp = self
            .device
            .create_buffer_from_fn::<f32>(PAR_REDUCE_BLOCK_SIZE, |_| 0.0)
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
        // a.values
        //     .copy_to_vec()
        //     .par_iter()
        //     .map(|x| x.abs())
        //     .max_by(|a, b| a.partial_cmp(b).unwrap())
        //     .unwrap()
    }
    pub fn add_scaled(&self, k: f32, a: &Grid<f32>, out: &Grid<f32>) {
        assert_eq!(a.res, out.res);
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
        match self.settings.preconditioner {
            Preconditioner::Identity => {
                self.pcg_r.values.copy_to_buffer(&self.pcg_z.values);
            }
            Preconditioner::IncompletePoisson => {
                self.incomplete_poission_kernel
                    .as_ref()
                    .unwrap()
                    .dispatch(self.pcg_r.res)
                    .unwrap();
            }
            _ => todo!(),
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

        let my_solution = self.p.values.copy_to_vec();
        self.build_A_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
        pcgsolver::eigen_solve(
            self.A.as_ref().unwrap(),
            &self.div_velocity.values,
            &self.p.values,
        );
        let eigen_solution = self.p.values.copy_to_vec();
        dbg!(&my_solution);
        dbg!(&eigen_solution);
        for (a, b) in my_solution.iter().zip(eigen_solution.iter()) {
            let a = *a;
            let b = *b;
            if (a - b).abs() / (b.abs() + 0.01) > 1e-3 {
                dbg!(a);
                dbg!(b);
                panic!("solver failed");
            }
        }
    }
    pub fn pressure_pcg_solver(&self) -> Option<u32> {
        let settings = self.control.copy_to_vec()[0];
        self.div_velocity.values.copy_to_buffer(&self.pcg_r.values);
        // let div_v = self.div_velocity.values.copy_to_vec();
        // dbg!(&div_v);
        // dbg!(div_v.len());
        self.r_eq_r_sub_A_x();
        let residual = self.abs_max(&self.pcg_r);
        dbg!(residual);
        if residual < settings.tolerance {
            return Some(0);
        }
        let tol = settings.tolerance * residual;
        self.apply_preconditioner();
        let mut rho = self.dot(&self.pcg_r, &self.pcg_z);
        dbg!(rho);
        if rho == 0.0 || rho.is_nan() {
            return None;
        }

        self.pcg_z.values.copy_to_buffer(&self.pcg_s.values);
        for i in 0..settings.max_iterations {
            self.z_eq_A_s();
            let alpha = rho / self.dot(&self.pcg_s, &self.pcg_z);
            self.add_scaled(alpha, &self.pcg_s, &self.p);
            self.add_scaled(-alpha, &self.pcg_z, &self.pcg_r);
            let residual = self.abs_max(&self.pcg_r);
            dbg!(i, residual);
            if residual < tol {
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

        self.particle_to_grid_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.u.res)
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
    fn transfter_grid_to_particles(&self) {
        self.grid_to_particle_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();
    }
    pub fn step(&self) {
        let settings = self.control.copy_to_vec()[0];
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
