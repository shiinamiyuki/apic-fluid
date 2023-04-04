use crate::{
    grid::Grid,
    pcgsolver::{PcgSolver, Preconditioner, Stencil},
    *,
};

#[derive(Clone, Copy, Value, Default, Debug)]
#[repr(C)]
pub struct Particle {
    pub pos: Float3,
    pub vel: Float3,
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
pub struct State {
    pub h: f32,
    // modified for CFL condition
    pub dt: f32,
    pub max_dt: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
    pub rho: f32,
}
#[derive(Clone, Copy)]
pub struct SimulationSettings {
    pub dt: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
    pub res: [u32; 3],
    pub h: f32,
    pub rho: f32,
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

    // fluid mass center at velocity samples
    pub mass_u: Grid<f32>,
    pub mass_v: Grid<f32>,
    pub mass_w: Grid<f32>,

    pub tmp_u: Grid<f32>,
    pub tmp_v: Grid<f32>,
    pub tmp_w: Grid<f32>,

    pub liquid_phi: Grid<f32>,

    pub p: Grid<f32>,

    pub rhs: Grid<f32>,

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
    build_linear_system_kernel: Option<Kernel<(f32,)>>,
    compute_phi_kernel: Option<Kernel<()>>,
    build_particle_list_kernel: Option<Kernel<()>>,
    particle_to_grid_kernel: Option<Kernel<()>>,
    grid_to_particle_kernel: Option<Kernel<()>>,
    enforce_boundary_kernel: Option<Kernel<()>>,
    velocity_update_kernel: Option<Kernel<(f32,)>>,
    compute_cfl_kernel: Option<Kernel<(Buffer<f32>,)>>,
    fluid_mass_kernel: Option<Kernel<()>>,
    zero_kernel: Kernel<(Buffer<f32>,)>,
    pub device: Device,
    state: State,
    cfl_tmp: Buffer<f32>,
}
#[derive(Clone, Copy, PartialOrd, PartialEq, Debug)]
pub enum VelocityIntegration {
    Euler,
    RK2,
    RK3,
}

impl Simulation {
    fn stencil_offsets(&self, i: u32) -> Expr<Int3> {
        let offset = make_int3(0, 1, -1);
        match i {
            0 => offset.xxx(),
            1 => offset.yxx(),
            2 => offset.zxx(),
            3 => offset.xyx(),
            4 => offset.xzx(),
            5 => offset.xxy(),
            6 => offset.xxz(),
            _ => panic!("invalid stencil offset index"),
        }
    }
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

        let mass_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let mass_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let mass_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

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
            mass_u,
            mass_v,
            mass_w,
            tmp_u,
            tmp_v,
            tmp_w,
            p,
            liquid_phi,
            solver: None,
            rhs: div_velocity,
            dimension,
            res,
            A: None,
            compute_phi_kernel: None,
            build_linear_system_kernel: None,
            particles_vec: Vec::new(),
            particles: None,
            apply_gravity_kernel: None,
            advect_particle_kernel: None,
            build_particle_list_kernel: None,
            particle_to_grid_kernel: None,
            grid_to_particle_kernel: None,
            velocity_update_kernel: None,
            compute_cfl_kernel: None,
            fluid_mass_kernel: None,
            enforce_boundary_kernel: None,
            cfl_tmp: device.create_buffer(1024).unwrap(),
            device: device.clone(),
            zero_kernel,
            state: State {
                h,
                rho: settings.rho,
                max_dt: dt,
                dt,
                max_iterations: settings.max_iterations,
                tolerance: settings.tolerance,
            },
        }
    }
    pub fn h(&self) -> Expr<f32> {
        // self.control.var().read(0).h()
        const_(self.state.h)
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
            cpu_dbg!(make_float3(u, v, w));
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
                    for axis in 0..self.dimension {
                        self.transfer_particles_to_grid_impl(
                            &self.p,
                            particles,
                            cell_idx,
                            axis,
                            pic_weight,
                            flip_weight,
                            apic_weight,
                        );
                    }
                    // let map = |g: &Grid<f32>, axis: u8| {
                    //     if_!(!g.oob(cell_idx.int()), {
                    //         self.transfer_particles_to_grid_impl(
                    //             g,
                    //             particles,
                    //             cell_idx,
                    //             axis,
                    //             pic_weight,
                    //             flip_weight,
                    //             apic_weight,
                    //         )
                    //     });
                    // };
                    // map(&self.u, 0);
                    // map(&self.v, 1);
                    // if self.dimension == 3 {
                    //     map(&self.w, 2);
                    // }
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
                    let phi = var!(f32, self.h());
                    let node_pos = self.p.pos_i_to_f(node.int());
                    let count = var!(u32, 0);
                    self.p
                        .for_each_particle_in_neighbor(node, [-1, -1, -1], [1, 1, 1], |pt_idx| {
                            let pt = particles.var().read(pt_idx);
                            let new_phi = pt.pos().distance(node_pos) - pt.radius();
                            // cpu_dbg!(pt.pos().distance(node_pos));
                            phi.store(phi.load().min(new_phi));
                            count.store(count.load() + 1);
                        });
                    // cpu_dbg!(count.load());
                    // cpu_dbg!(make_uint4(node.x(), node.y(), node.z(), count.load()));
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
                    let new_pos = pt.pos() + pt.vel() * dt;
                    let lo = Float3Expr::zero();
                    let hi = make_uint3(self.res[0], self.res[1], self.res[2]).float() * self.h();
                    let new_pos = new_pos.clamp(lo - 0.5 * self.h(), hi - self.h() * 0.5);
                    let pt = pt.set_pos(new_pos);
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
        self.build_linear_system_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| {
                    self.build_linear_system_impl(dt);
                })
                .unwrap(),
        );
        self.fluid_mass_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.compute_fluid_mass_impl();
                })
                .unwrap(),
        );
        self.velocity_update_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| self.update_velocity_impl(dt))
                .unwrap(),
        );
        self.enforce_boundary_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    let map = |g: &Grid<f32>, axis: usize| {
                        self.enforce_boundary_impl(g, axis);
                    };
                    map(&self.u, 0);
                    map(&self.v, 1);
                    if self.dimension == 3 {
                        map(&self.w, 2);
                    }
                })
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
        axis: usize,
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
        let apic_mv = var!(f32); // momentum

        let m = var!(f32); // mass
        let (lo, hi) = match axis {
            0 => ([-1, -1, -1], [0, 1, 1]),
            1 => ([-1, -1, -1], [1, 0, 1]),
            2 => ([-1, -1, -1], [1, 1, 0]),
            _ => unreachable!(),
        };
        g.for_each_particle_in_neighbor(cell_idx, lo, hi, |pt_idx| {
            let pt = particles.var().read(pt_idx);
            let v_p = pt.vel();
            let offset = (pt.pos() - cell_pos) / g.dx;
            // if_!(!offset.abs().cmple(1.001).all(), {
            //     cpu_dbg!(offset);
            // });
            // assert(offset.abs().cmple(1.001).all());
            let w_p = trilinear_weight(offset, self.dimension);
            let m_p = 1.0;
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

            pic_flip_mv.store(pic_flip_mv.load() + m_p * v_pa * w_p);

            apic_mv.store(apic_mv.load() + m_p * w_p * (v_pa + c_pa.dot(cell_pos - pt.pos())));
            m.store(m.load() + w_p * m_p);
        });
        let m = select(m.load().cmpeq(0.0), const_(1.0f32), m.load());
        let pic_flip_v = pic_flip_mv.load() / m;
        let apic_v = apic_mv.load() / m;
        // TODO: check this
        let final_v = (pic_weight + flip_weight) * pic_flip_v + apic_weight * apic_v;
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
        let v_pa = v_p.at(axis as usize);
        g.for_each_neighbor_node(pt.pos(), |node| {
            let node_pos = g.pos_i_to_f(node.int());
            let v_i = g.at_index(node);
            let old_v_i = old_g.at_index(node);
            let diff_v = v_i - old_v_i;
            let offset = (pt.pos() - node_pos) / g.dx;
            // cpu_dbg!(offset);
            // assert(offset.abs().cmple(1.0).all());
            // assert(offset.abs().cmpge(0.0).all());
            // cpu_dbg!(pt.pos());
            // cpu_dbg!(node_pos);
            let w_pa = trilinear_weight(offset, self.dimension);
            // TODO: should i divide dx?
            let grad_w_pa = grad_trilinear_weight(offset, self.dimension) / g.dx;

            flip_v_pa.store(flip_v_pa.load() + diff_v * w_pa);
            pic_v_pa.store(pic_v_pa.load() + v_i * w_pa);
            apic_c_pa.store(apic_c_pa.load() + grad_w_pa * v_i);
        });
        // TODO: check this
        let v_pa =
            (apic_weight + pic_weight) * pic_v_pa.load() + flip_weight * (v_pa + flip_v_pa.load());
        let pt = var!(Particle, pt);

        match axis {
            0 => {
                pt.set_vel(v_p.set_x(v_pa));
                pt.set_c_x(apic_c_pa);
            }
            1 => {
                // cpu_dbg!(v_pa);
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
    fn compute_fluid_mass_impl(&self) {
        let x = dispatch_id();
        let unit_mass = self.state.rho
            * if self.dimension == 2 {
                self.h() * self.h() * self.state.rho
            } else {
                self.h() * self.h() * self.h() * self.state.rho
            };
        let map = |u: &Grid<f32>, mass_u: &Grid<f32>, axis: usize| {
            if_!(!u.oob(x.int()), {
                let x_a = x.at(axis);
                let mass = if_!(x_a.cmpeq(0) | x_a.cmpeq(u.res[axis] - 1), {
                    const_(0.0f32)
                }, else {
                    if_!(x.cmpeq(0).any() | x.cmpeq(make_uint3(u.res[0], u.res[1], u.res[2]) - 1).any(), {
                        const_(0.5f32)
                    }, else {
                        const_(1.0f32)
                    })
                });
                mass_u.set_index(x, mass * unit_mass);
            });
        };
        map(&self.u, &self.mass_u, 0);
        map(&self.v, &self.mass_v, 1);
        if self.dimension == 3 {
            map(&self.w, &self.mass_w, 2);
        }
    }
    fn build_linear_system_impl(&self, dt: Expr<f32>) {
        let x = dispatch_id();
        let i = self.p.linear_index(x);
        let A = self.A.as_ref().unwrap();
        let A_coeff = A.coeff.var();
        let h = self.h();
        let rho = self.state.rho;
        let rho2 = rho * rho;
        let h2 = h * h;
        let diag = var!(f32);
        let phi_x = self.liquid_phi.at_index(x);

        // get fluid mass around current pressure sample
        let get_fluid_mass = |offset_idx: u32| {
            match offset_idx {
                0 => panic!("don't call 0 directly"),
                1 => {
                    // x+1/2
                    self.mass_u.at_index(x + make_uint3(1, 0, 0))
                }
                2 => {
                    // x-1/2
                    self.mass_u.at_index(x)
                }
                3 => {
                    // y+1/2
                    self.mass_v.at_index(x + make_uint3(0, 1, 0))
                }
                4 => {
                    // y-1/2
                    self.mass_v.at_index(x)
                }
                5 => {
                    // z+1/2
                    self.mass_w.at_index(x + make_uint3(0, 0, 1))
                }
                6 => {
                    // z-1/2
                    self.mass_w.at_index(x)
                }
                _ => unreachable!(),
            }
        };
        let lhs_scale = dt / (h2 * rho2);
        // only build the system for fluid cells
        if_!(phi_x.cmplt(0.0), {
            let du = var!(f32);
            let dv = var!(f32);
            let dw = var!(f32);
            let compute_stencil = |offset_idx: u32| {
                let offset = A.offsets.var().read(offset_idx);
                let y = x.int() + offset;
                if_!(!self.liquid_phi.oob(y), {
                    let phi_y = self.liquid_phi.at_index(y.uint());
                    let mass = get_fluid_mass(offset_idx);
                    if_!(phi_y.cmpgt(0.0), {
                        let theta = self.free_surface_theta(phi_x, phi_y).max(1e-3);
                        assert(theta.cmpge(0.0) & theta.cmple(1.0));
                        // 1 + (1 - theta) / theta = 1 / theta
                        diag.store(diag.load() + mass / theta * lhs_scale);
                        A_coeff.write(i * A.offsets.len() as u32 + offset_idx, 0.0);
                    }, else {
                        diag.store(diag.load() + mass * lhs_scale);
                        A_coeff.write(i * A.offsets.len() as u32 + offset_idx, -mass * lhs_scale);
                    });
                    //
                });
            };
            if self.dimension == 2 {
                assert_eq!(A.offsets.len(), 5);
                for j in 1..5 {
                    compute_stencil(j);
                }
            } else {
                assert_eq!(A.offsets.len(), 7);
                for j in 1..7 {
                    compute_stencil(j);
                }
            }
            A_coeff.write(i * A.offsets.len() as u32, diag.load());
            {
                let offset = make_uint2(0, 1);
                let weighted_velocity = |u: &Grid<f32>, mass_u: &Grid<f32>, x: Expr<Uint3>| {
                    u.at_index(x) * mass_u.at_index(x)
                };
                du.store(
                    weighted_velocity(&self.u, &self.mass_u, x + offset.yxx())
                        - weighted_velocity(&self.u, &self.mass_u, x),
                );
                dv.store(
                    weighted_velocity(&self.v, &self.mass_v, x + offset.xyx())
                        - weighted_velocity(&self.v, &self.mass_v, x),
                );
                if self.dimension == 3 {
                    dw.store(
                        weighted_velocity(&self.w, &self.mass_w, x + offset.xxy())
                            - weighted_velocity(&self.w, &self.mass_w, x),
                    );
                }
            }
            let div = if self.dimension == 2 {
                (-du.load() - dv.load()) / (self.state.rho * self.h())
            } else {
                (-du.load() - dv.load() - dw.load()) / (self.state.rho * self.h())
            };
            // cpu_dbg!(make_float3(du.load(), dv.load(), dw.load()));
            self.rhs.set_index(x, div);
        });
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
    fn copy_velocity(&self) {
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
    pub fn solve_pressure(&self, dt: f32) {
        self.copy_velocity();
        let solver = self.solver.as_ref().unwrap();
        let A = self.A.as_ref().unwrap();
        solver
            .zero
            .dispatch([A.coeff.len() as u32, 1, 1], &A.coeff)
            .unwrap();
        solver
            .zero
            .dispatch([self.rhs.values.len() as u32, 1, 1], &self.rhs.values)
            .unwrap();
        // dbg!(&self.rhs.values.copy_to_vec());
        solver
            .zero
            .dispatch([self.p.values.len() as u32, 1, 1], &self.p.values)
            .unwrap();
        self.build_linear_system_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res, &dt)
            .unwrap();
        // dbg!(&self.v.values.copy_to_vec());
        // dbg!(&A.coeff.copy_to_vec());
        // dbg!(&self.rhs.values.copy_to_vec());
        let i = solver.solve(self.A.as_ref().unwrap(), &self.rhs.values, &self.p.values);
        if i.is_none() {
            log::warn!("pressure solver failed");
        } else {
            log::info!("pressure solve finished in {} iterations", i.unwrap());
        }

        // pcgsolver::eigen_solve(self.A.as_ref().unwrap(), &self.rhs.values, &self.p.values);

        self.velocity_update_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1, &dt)
            .unwrap();
        // dbg!(&self.v.values.copy_to_vec());
    }
    fn transfer_particles_to_grid(&self) {
        self.p.reset_particle_list();
        self.build_particle_list_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();

        self.particle_to_grid_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
    }
    fn transfer_grid_to_particles(&self) {
        self.grid_to_particle_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();

        // dbg!(self.particles.as_ref().unwrap().copy_to_vec());
    }
    fn compute_liquid_phi(&self) {
        self.compute_phi_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
        // dbg!(self.liquid_phi.values.copy_to_vec());
        let phi = self.liquid_phi.values.copy_to_vec();
        let non_empty = phi.iter().filter(|&&x| x < 0.0).count();
        log::info!("non empty cells: {}", non_empty);
    }
    fn compute_fluid_mass(&self) {
        self.fluid_mass_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1)
            .unwrap();
    }
    pub fn step(&mut self) {
        let mut cur = 0.0f32;
        while cur < self.state.max_dt {
            profile("compute_cfl", || {
                self.compute_cfl();
            });

            let dt = self.state.dt;
            log::info!("dt: {}", dt);
            profile("advect_particle", || {
                self.advect_particle(dt);
            });
            profile("transfer_particles_to_grid", || {
                self.transfer_particles_to_grid();
            });
            profile("compute_liquid_phi", || {
                self.compute_liquid_phi();
            });
            profile("compute_fluid_mass", || {
                self.compute_fluid_mass();
            });
            profile("apply_ext_forces", || {
                self.apply_ext_forces(dt);
                self.enforce_boundary();
            });
            profile("solve_pressure", || {
                self.solve_pressure(dt);
                self.enforce_boundary();
            });
            profile("transfer_grid_to_particles", || {
                self.transfer_grid_to_particles();
            });
            cur += dt;
        }
    }
    // solve for (1 - theta) * cur_phi + theta * neighbor_phi = 0
    //  cur_phi - theta * cur_phi + theta * neighbor_phi = 0
    //  -theta * (cur_phi - neighbor_phi) = -cur_phi
    //  theta = -cur_phi / (cur_phi - neighbor_phi)
    fn free_surface_theta(&self, cur_phi: Expr<f32>, neighbor_phi: Expr<f32>) -> Expr<f32> {
        assert(cur_phi.cmplt(0.0));
        cur_phi / (neighbor_phi - cur_phi)
    }
    fn enforce_boundary_impl(&self, g: &Grid<f32>, axis: usize) {
        let x = dispatch_id();
        let x_a = x.at(axis);
        if_!(!g.oob(x.int()), {
            if_!(x_a.cmpeq(0) | x_a.cmpeq(g.res[axis] - 1), {
                g.set_index(x, 0.0.into())
            })
        });
    }

    // Make sure the velocity is zero on grid boundaries
    fn enforce_boundary(&self) {
        self.enforce_boundary_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1)
            .unwrap();
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
                    let u_cur = u.at_index(i.uint());
                    // let phi_left = self.liquid_phi.at_index((i - off).uint());
                    // let phi_right = self.liquid_phi.at_index(i.uint());
                    // let both_in_fluid = phi_left.cmplt(0.0) & phi_right.cmplt(0.0);
                    // let both_in_air = phi_left.cmpge(0.0) & phi_right.cmpge(0.0);
                    // if_!(both_in_fluid, {
                    //     let grad_pressure =
                    //         (self.p.at_index(i.uint()) - self.p.at_index((i - off).uint())) / self.h();
                    //     // let grad_pressure =
                    //     //     (self.p.at_index((i + off).uint()) - self.p.at_index(i.uint())) / self.h();

                    //     u.set_index(i.uint(), u_cur - dt * grad_pressure / self.state.rho);
                    // }, else {
                    //     if_!(!both_in_air, {
                    //         if_!(phi_left.cmplt(0.0),{
                    //             let theta = self.free_surface_theta(phi_left, phi_right);
                    //             let grad_pressure = (self.p.at_index(i.uint())  /  theta) / self.h();
                    //             u.set_index(i.uint(), u_cur - dt * grad_pressure / self.state.rho);
                    //         }, else {
                    //             let theta = self.free_surface_theta(phi_right, phi_left);
                    //             let grad_pressure = (- self.p.at_index((i - off).uint())  /  theta) / self.h();
                    //             u.set_index(i.uint(), u_cur - dt * grad_pressure / self.state.rho);
                    //         });
                    //     }, else {
                    //         u.set_index(i.uint(), 0.0.into());
                    //     });
                    // });
                    let grad_pressure =
                            (self.p.at_index(i.uint()) - self.p.at_index((i - off).uint())) / self.h();
                        // let grad_pressure =
                        //     (self.p.at_index((i + off).uint()) - self.p.at_index(i.uint())) / self.h();

                    u.set_index(i.uint(), u_cur - dt * grad_pressure / self.state.rho);
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
    fn compute_cfl(&mut self) {
        let s = self.device.default_stream();
        s.with_scope(|s| {
            s.submit([
                self.zero_kernel
                    .dispatch_async([self.cfl_tmp.len() as u32, 1, 1], &self.cfl_tmp),
                self.compute_cfl_kernel
                    .as_ref()
                    .unwrap()
                    .dispatch_async([self.particles_vec.len() as u32, 1, 1], &self.cfl_tmp),
            ])
            .unwrap()
        });
        let max_vel = self
            .cfl_tmp
            .copy_to_vec()
            .iter()
            .fold(0.0f32, |a, b| a.max(*b));
        self.state.dt = (self.state.h / (max_vel + 1e-6)).clamp(1e-4, self.state.max_dt);
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
pub(crate) fn profile(name: &str, f: impl FnOnce()) {
    let t0 = std::time::Instant::now();
    f();
    let elapsed = (std::time::Instant::now() - t0).as_millis();
    log::info!("[{}] finished in {}ms", name, elapsed);
}
