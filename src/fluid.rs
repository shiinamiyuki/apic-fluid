use luisa::{
    rtx::{Accel, Mesh},
    AccelBuildRequest, AccelOption,
};

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
    pub density: f32, // assuming all particles have same volume

    pub c_x: Float3,
    pub c_y: Float3,
    pub c_z: Float3,

    pub tag: u32,
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
}
#[derive(Clone, Copy)]
pub struct SimulationSettings {
    pub dt: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
    pub res: [u32; 3],
    pub h: f32,
    pub g: f32,
    pub dimension: usize,
    pub transfer: ParticleTransfer,
    pub force_wall_separation: bool,
    pub seperation_threshold: f32,
    pub advect: VelocityIntegration,
    pub preconditioner: Preconditioner,
}
pub struct Simulation {
    pub settings: SimulationSettings,
    res_p1: [u32; 3],

    pub u: Grid<f32>,
    pub v: Grid<f32>,
    pub w: Grid<f32>,

    pub density_u: Grid<f32>,
    pub density_v: Grid<f32>,
    pub density_w: Grid<f32>,

    // fluid mass center at velocity samples
    pub mass_u: Grid<f32>,
    pub mass_v: Grid<f32>,
    pub mass_w: Grid<f32>,

    // the volume around velocity samples
    // computed from solid wall biundaries
    pub static_mass_u: Grid<f32>,
    pub static_mass_v: Grid<f32>,
    pub static_mass_w: Grid<f32>,

    pub tmp_u: Grid<f32>,
    pub tmp_v: Grid<f32>,
    pub tmp_w: Grid<f32>,

    pub has_value_u: Grid<bool>,
    pub has_value_v: Grid<bool>,
    pub has_value_w: Grid<bool>,

    pub tmp_has_value_u: Grid<bool>,
    pub tmp_has_value_v: Grid<bool>,
    pub tmp_has_value_w: Grid<bool>,

    pub fluid_phi: Grid<f32>,

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
    static_fluid_mass_kernel: Option<Kernel<()>>,
    fluid_mass_kernel: Option<Kernel<()>>,
    extrapolate_velocity_kernel: Option<Kernel<()>>,
    extrapolate_density_kernel: Option<Kernel<()>>,
    zero_kernel: Kernel<(Buffer<f32>,)>,
    pub device: Device,
    state: State,
    cfl_tmp: Buffer<f32>,
    accel: Option<Accel>,
    mesh: Option<(Buffer<Float3>, Buffer<Uint3>)>,
    pub log_volume: bool,
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

        let density_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let density_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let density_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let mass_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let mass_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let mass_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let static_mass_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let static_mass_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let static_mass_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let tmp_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let tmp_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let tmp_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let has_value_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let has_value_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let has_value_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let tmp_has_value_u = Grid::new(device.clone(), u_res, dimension, [-0.5 * h, 0.0, 0.0], h);
        let tmp_has_value_v = Grid::new(device.clone(), v_res, dimension, [0.0, -0.5 * h, 0.0], h);
        let tmp_has_value_w = Grid::new(device.clone(), w_res, dimension, [0.0, 0.0, -0.5 * h], h);

        let make_pressure_grid = || {
            Grid::new(
                device.clone(),
                [res[0], res[1], if dimension == 3 { res[2] } else { 1 }], // boundary is zero
                dimension,
                [0.0, 0.0, 0.0],
                h,
            )
        };
        let fluid_phi = make_pressure_grid();

        let p = make_pressure_grid();
        // let pcg_tmp = make_pressure_grid();
        let div_velocity = make_pressure_grid();
        let zero_kernel = device
            .create_kernel_async::<(Buffer<f32>,)>(&|buf: BufferVar<f32>| {
                set_block_size([512, 1, 1]);
                let i = dispatch_id().x();
                buf.write(i, 0.0);
            })
            .unwrap();
        Self {
            log_volume: true,
            res_p1: [
                res[0] + 1,
                res[1] + 1,
                if dimension == 3 { res[2] + 1 } else { 1 },
            ],
            settings,
            u,
            v,
            w,
            density_u,
            density_v,
            density_w,
            mass_u,
            mass_v,
            mass_w,
            static_mass_u,
            static_mass_v,
            static_mass_w,
            tmp_u,
            tmp_v,
            tmp_w,
            has_value_u,
            has_value_v,
            has_value_w,
            tmp_has_value_u,
            tmp_has_value_v,
            tmp_has_value_w,
            p,
            fluid_phi,
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
            extrapolate_velocity_kernel: None,
            extrapolate_density_kernel: None,
            enforce_boundary_kernel: None,
            static_fluid_mass_kernel: None,
            cfl_tmp: device.create_buffer(1024).unwrap(),
            device: device.clone(),
            zero_kernel,
            accel: None,
            mesh: None,
            state: State {
                h,
                max_dt: dt,
                dt,
                max_iterations: settings.max_iterations,
                tolerance: settings.tolerance,
            },
        }
    }
    pub fn set_mesh(&mut self, vertices: Buffer<Float3>, faces: Buffer<Uint3>) {
        let mesh = self
            .device
            .create_mesh(vertices.view(..), faces.view(..), AccelOption::default())
            .unwrap();
        mesh.build(AccelBuildRequest::ForceBuild);
        let accel = self.device.create_accel(Default::default()).unwrap();
        accel.push_mesh(&mesh, Mat4::identity(), u8::MAX, true);
        accel.build(AccelBuildRequest::ForceBuild);
        self.accel = Some(accel);
        self.mesh = Some((vertices, faces));
    }
    fn signed_distance(&self, ray: Expr<RtxRay>) -> Expr<f32> {
        let accel = self.accel.as_ref().unwrap();
        let accel = accel.var();
        let hit = accel.trace_closest(ray);
        if_!(hit.valid(), {
            let prim_id = hit.prim_id();
            let u = hit.u();
            let v = hit.v();
            let (vertices, faces) = self.mesh.as_ref().unwrap();
            let vertices = vertices.var();
            let faces = faces.var();
            let f = faces.read(prim_id);
            let v0 = vertices.read(f.x());
            let v1 = vertices.read(f.y());
            let v2 = vertices.read(f.z());
            // let n = (v0 - v1).cross(v0 - v2).normalize();
            let n = (v2 - v0).cross(v1 - v0).normalize();
            let p = (1.0 - u - v) * v0 + u * v1 + v * v2;
            let dist = (p - make_float3(ray.orig_x(), ray.orig_y(), ray.orig_z())).length();
            let d = make_float3(ray.dir_x(), ray.dir_y(), ray.dir_z());
            let inside = d.dot(n).cmplt(0.0);
            dist * select(inside, const_(-1.0f32), const_(1.0f32))
        }, else {
            const_(1e5f32)
        })
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
                    set_block_size([256, 1, 1]);
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
                    let map = |g: &Grid<f32>, density_g: &Grid<f32>, axis: usize| {
                        if_!(!g.oob(cell_idx.int()), {
                            self.transfer_particles_to_grid_impl(
                                g, density_g, particles, cell_idx, axis, transfer,
                            )
                        });
                    };
                    map(&self.u, &self.density_u, 0);
                    map(&self.v, &self.density_v, 1);
                    if self.dimension == 3 {
                        map(&self.w, &self.density_w, 2);
                    }
                })
                .unwrap(),
        );
        self.grid_to_particle_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    set_block_size([256, 1, 1]);
                    let particles = self.particles.as_ref().unwrap();
                    let pt_idx = dispatch_id().x();
                    let transfer = self.settings.transfer;
                    let map =
                        |g: &Grid<f32>, old_g: &Grid<f32>, has_value: &Grid<bool>, axis: u8| {
                            self.transfer_grid_to_particles_impl(
                                g, old_g, has_value, particles, pt_idx, axis, transfer,
                            )
                        };
                    map(&self.u, &self.tmp_u, &self.has_value_u, 0);
                    map(&self.v, &self.tmp_v, &self.has_value_v, 1);
                    if self.dimension == 3 {
                        map(&self.w, &self.tmp_w, &self.has_value_w, 2);
                    }
                })
                .unwrap(),
        );
        self.compute_phi_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    let node = dispatch_id();
                    let particles = self.particles.as_ref().unwrap();
                    let phi = var!(f32, self.h() * (self.dimension as f32).sqrt());
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
                    self.fluid_phi.set_index(node, phi.load());
                })
                .unwrap(),
        );
        self.advect_particle_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| {
                    set_block_size([256, 1, 1]);
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap();
                    let pt = particles.var().read(i);
                    let new_pos = pt.pos() + pt.vel() * dt;
                    let lo = Float3Expr::zero();
                    let hi =
                        (make_uint3(self.res[0], self.res[1], self.res[2]) - 1).float() * self.h();
                    // let new_pos = new_pos.clamp(lo - 0.5 * self.h(), hi + self.h() * 0.498);
                    // let new_pos = new_pos.clamp(lo, hi);
                    let vel = var!(Float3, pt.vel());
                    if self.dimension == 2 {
                        vel.store(vel.load().set_z(0.0.into()));
                    }
                    for axis in 0..3 {
                        let p_a = var!(f32, new_pos.at(axis));
                        let v_a = var!(f32, vel.load().at(axis));
                        if_!(p_a.load().cmplt(lo.at(axis)), {
                            v_a.store(v_a.load().max(0.0));
                        });
                        if_!(p_a.load().cmpgt(hi.at(axis)), {
                            v_a.store(v_a.load().min(0.0));
                        });
                        let v_a = v_a.load();
                        match axis {
                            0 => {
                                vel.store(vel.load().set_x(v_a));
                            }
                            1 => {
                                vel.store(vel.load().set_y(v_a));
                            }
                            2 => {
                                vel.store(vel.load().set_z(v_a));
                            }
                            _ => unreachable!(),
                        }
                    }
                    let new_pos = new_pos.clamp(lo, hi);
                    let pt = pt.set_pos(new_pos).set_vel(vel.load());
                    particles.var().write(i, pt);
                })
                .unwrap(),
        );
        self.apply_gravity_kernel = Some(
            self.device
                .create_kernel_async::<(f32,)>(&|dt: Expr<f32>| {
                    let p = dispatch_id();
                    let v = self.v.at_index(p);
                    self.v.set_index(p, v - self.settings.g * dt);
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
                    set_block_size([256, 1, 1]);
                    let i = dispatch_id().x();
                    let particles = self.particles.as_ref().unwrap().var();
                    let pt = particles.read(i);
                    let v = pt.vel().length();
                    tmp.atomic_fetch_max(i % tmp.len(), v);
                })
                .unwrap(),
        );
        self.extrapolate_velocity_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.extrapolate_velocity_impl();
                })
                .unwrap(),
        );
        self.extrapolate_density_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.extrapolate_density_impl();
                })
                .unwrap(),
        );
        self.static_fluid_mass_kernel = Some(
            self.device
                .create_kernel_async::<()>(&|| {
                    self.compute_static_fluid_mass_impl();
                })
                .unwrap(),
        );
        self.static_fluid_mass_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1)
            .unwrap();
    }
    pub fn transfer_particles_to_grid_impl(
        &self,
        g: &Grid<f32>,
        density_g: &Grid<f32>,
        particles: &Buffer<Particle>,
        cell_idx: Expr<Uint3>,
        axis: usize,
        transfer: ParticleTransfer,
    ) {
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
        let sum_w = var!(f32, 0.0);
        self.p
            .for_each_particle_in_neighbor(cell_idx, lo, hi, |pt_idx| {
                let pt = particles.var().read(pt_idx);
                let v_p = pt.vel();
                let offset = (pt.pos() - cell_pos) / g.dx;
                // if_!(!offset.abs().cmple(1.001).all(), {
                //     cpu_dbg!(offset);
                // });
                // assert(offset.abs().cmple(1.001).all());
                let w_p = trilinear_weight(offset, self.dimension);
                // cpu_dbg!(w_p);
                assert(w_p.cmpge(0.0) & w_p.cmple(1.0));
                let m_p = pt.density();
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
                sum_w.store(sum_w.load() + w_p);
            });
        let m = m.load();
        let pic_flip_v = pic_flip_mv.load() / select(m.cmpeq(0.0), const_(1.0f32), m);
        let apic_v = apic_mv.load() / select(m.cmpeq(0.0), const_(1.0f32), m);
        let final_v = match transfer {
            ParticleTransfer::Pic | ParticleTransfer::Flip | ParticleTransfer::PicFlip(_) => {
                pic_flip_v
            }
            ParticleTransfer::Apic => apic_v,
        };
        density_g.set_index(
            cell_idx,
            m / select(sum_w.load().cmpeq(0.0), const_(1.0f32), sum_w.load()),
        );
        g.set_index(cell_idx, final_v);
    }
    fn transfer_grid_to_particles_impl(
        &self,
        g: &Grid<f32>,
        old_g: &Grid<f32>,
        has_value: &Grid<bool>,
        particles: &Buffer<Particle>,
        pt_idx: Expr<u32>,
        axis: u8,
        transfer: ParticleTransfer,
    ) {
        let particles = particles.var();
        let pt = particles.read(pt_idx);

        let v_p = pt.vel();

        let pic_v_pa = var!(f32);
        let flip_v_pa = var!(f32);

        let apic_c_pa = var!(Float3);
        let v_pa = v_p.at(axis as usize);
        g.for_each_neighbor_node(pt.pos(), |node| {
            if_!(has_value.at_index(node), {
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
                assert(w_pa.cmpge(0.0) & w_pa.cmple(1.0));
                // TODO: should i divide dx?
                let grad_w_pa = grad_trilinear_weight(offset, self.dimension) / g.dx;

                flip_v_pa.store(flip_v_pa.load() + diff_v * w_pa);
                pic_v_pa.store(pic_v_pa.load() + v_i * w_pa);
                apic_c_pa.store(apic_c_pa.load() + grad_w_pa * v_i);
            });
        });
        let v_pa = match transfer {
            ParticleTransfer::Pic => pic_v_pa.load(),
            ParticleTransfer::Flip => flip_v_pa.load() + v_pa,
            ParticleTransfer::PicFlip(flip_ratio) => {
                pic_v_pa.load() * (1.0 - flip_ratio) + (flip_v_pa.load() + v_pa) * flip_ratio
            }
            ParticleTransfer::Apic => pic_v_pa.load(),
        };
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
    // fn unit_mass(&self) -> Expr<f32> {
    //     self.state.rho
    //         * if self.dimension == 2 {
    //             self.h() * self.h() * self.state.rho
    //         } else {
    //             self.h() * self.h() * self.h() * self.state.rho
    //         }
    // }
    fn compute_static_fluid_mass_impl(&self) {
        let x = dispatch_id();

        let map = |u: &Grid<f32>, mass_u: &Grid<f32>, axis: usize| {
            if_!(!u.oob(x.int()), {
                let x_a = x.at(axis);
                let mass = if_!(x_a.cmpeq(0) | x_a.cmpeq(u.res[axis] - 1), {
                    const_(0.0f32)
                }, else {
                    if self.accel.is_some() {
                        let ro = u.pos_i_to_f(x.int()) + make_float3(0.0, 0.0, 0.0);
                        let rd = match axis {
                            0=>make_float3(1.0,0.0,0.0),
                            1=>make_float3(0.0,1.0,0.0),
                            2=>make_float3(0.0,0.0,1.0),
                            _=>unreachable!()
                        };
                        let ray0 = RtxRayExpr::new(ro.x(), ro.y(), ro.z(), 0.0, rd.x(), rd.y(), rd.z(), 1e5);
                        let sd0 = self.signed_distance(ray0).clamp(0.0, 0.5 * self.h());

                        let ray1 = RtxRayExpr::new(ro.x(), ro.y(), ro.z(), 0.0, -rd.x(), -rd.y(), -rd.z(), 1e5);
                        let sd1 = self.signed_distance(ray1).clamp(0.0, 0.5 * self.h());
                        (sd0 + sd1) / self.h()
                    }else {
                        const_(1.0f32)
                    }
                });
                mass_u.set_index(x, mass);
                // has_value_u.set_index(x, true.into());
            });
        };
        map(&self.u, &self.static_mass_u, 0);
        map(&self.v, &self.static_mass_v, 1);
        if self.dimension == 3 {
            map(&self.w, &self.static_mass_w, 2);
        }
    }
    fn compute_fluid_mass_impl(&self) {
        let x = dispatch_id();

        let map = |u: &Grid<f32>,
                   density_u: &Grid<f32>,
                   static_mass_u: &Grid<f32>,
                   mass_u: &Grid<f32>,
                   has_value_u: &Grid<bool>,
                   axis: usize| {
            if_!(!u.oob(x.int()), {
                let rho = density_u.at_index(x);
                let unit_mass = rho
                    * if self.dimension == 2 {
                        self.h() * self.h()
                    } else {
                        self.h() * self.h() * self.h()
                    };
                let off = match axis {
                    0 => make_int3(1, 0, 0),
                    1 => make_int3(0, 1, 0),
                    2 => make_int3(0, 0, 1),
                    _ => unreachable!(),
                };
                let x_a = x.at(axis);
                let mass = static_mass_u.at_index(x);
                mass_u.set_index(x, mass * unit_mass);
                if_!(x_a.cmpeq(0) | x_a.cmpeq(u.res[axis] - 1), {
                    has_value_u.set_index(x, false.into());
                }, else{
                    let phi_left = self.fluid_phi.at_index((x.int() - off).uint());
                    let phi_right = self.fluid_phi.at_index(x.uint());
                    let both_in_air = phi_left.cmpgt(0.0) & phi_right.cmpgt(0.0);
                    has_value_u.set_index(x, !both_in_air & mass.cmpgt(0.0));
                });
                // has_value_u.set_index(x, true.into());
            });
        };
        map(
            &self.u,
            &self.density_u,
            &self.static_mass_u,
            &self.mass_u,
            &self.has_value_u,
            0,
        );
        map(
            &self.v,
            &self.density_v,
            &self.static_mass_v,
            &self.mass_v,
            &self.has_value_v,
            1,
        );
        if self.dimension == 3 {
            map(
                &self.w,
                &self.density_w,
                &self.static_mass_w,
                &self.mass_w,
                &self.has_value_w,
                2,
            );
        }
    }
    fn build_linear_system_impl(&self, dt: Expr<f32>) {
        let x = dispatch_id();
        let i = self.p.linear_index(x);
        let A = self.A.as_ref().unwrap();
        let A_coeff = A.coeff.var();
        let h = self.h();
        let h2 = h * h;
        let diag = var!(f32);
        let phi_x = self.fluid_phi.at_index(x);

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
        let get_fluid_rho = |offset_idx: u32| {
            match offset_idx {
                0 => panic!("don't call 0 directly"),
                1 => {
                    // x+1/2
                    self.density_u.at_index(x + make_uint3(1, 0, 0))
                }
                2 => {
                    // x-1/2
                    self.density_u.at_index(x)
                }
                3 => {
                    // y+1/2
                    self.density_v.at_index(x + make_uint3(0, 1, 0))
                }
                4 => {
                    // y-1/2
                    self.density_v.at_index(x)
                }
                5 => {
                    // z+1/2
                    self.density_w.at_index(x + make_uint3(0, 0, 1))
                }
                6 => {
                    // z-1/2
                    self.density_w.at_index(x)
                }
                _ => unreachable!(),
            }
        };
        // assert(rho.cmpgt(0.0));
        // cpu_dbg!(rho);

        // only build the system for fluid cells
        if_!(phi_x.cmplt(0.0), {
            // let off = make_uint2(0, 1);
            // cpu_dbg!(rho);
            // let rho2 = rho * rho;
            // let lhs_scale = dt / (h2 * rho2);
            // let rhs_scale = 1.0 / (h * rho);

            // assert(rho.cmpgt(0.0));
            let du = var!(f32);
            let dv = var!(f32);
            let dw = var!(f32);
            let compute_stencil = |offset_idx: u32| {
                let offset = A.offsets.var().read(offset_idx);
                let y = x.int() + offset;
                if_!(!self.fluid_phi.oob(y), {
                    let phi_y = self.fluid_phi.at_index(y.uint());
                    let mass = get_fluid_mass(offset_idx);
                    let rho = get_fluid_rho(offset_idx);
                    let rho2 = rho * rho;
                    let lhs_scale = dt / (h2 * rho2);

                    if_!(phi_y.cmpgt(0.0), {
                        assert(rho.cmpgt(0.0));
                        let theta = self.free_surface_theta(phi_x, phi_y); // 1e-3 would make fluid explode
                        assert(theta.is_finite());
                        assert(theta.cmpge(0.0) & theta.cmple(1.0));
                        let theta = theta.max(1e-2);
                        // 1 + (1 - theta) / theta = 1 / theta
                        diag.store(diag.load() + mass / theta * lhs_scale);
                        A_coeff.write(i * A.offsets.len() as u32 + offset_idx, 0.0);
                    }, else {
                        diag.store(diag.load() + mass * lhs_scale);
                        A_coeff.write(i * A.offsets.len() as u32 + offset_idx, -mass * lhs_scale);
                    });
                    //
                }, else {
                    // if velocity is leaving the boundary
                    // then treat the boundary as free surface

                    // if self.settings.force_wall_separation {
                    //     let is_free_surface = var!(bool, false);
                    //     let thr = self.settings.seperation_threshold * dt;
                    //     match offset_idx {
                    //         // 1=>{
                    //         //     is_free_surface.store(x.x().cmpeq(self.p.res[0]-1) & self.u.at_index(x+make_uint3(1,0,0)).cmplt(-thr));
                    //         //     if_!(is_free_surface.load(), {
                    //         //         du.store(du.load() + self.unit_mass() * self.u.at_index(x+make_uint3(1,0,0)));
                    //         //     });
                    //         // }
                    //         // 2=>{
                    //         //     is_free_surface.store(x.x().cmpeq(0) & self.u.at_index(x).cmpgt(thr));
                    //         //     if_!(is_free_surface.load(), {
                    //         //         du.store(du.load() - self.unit_mass() * self.u.at_index(x));
                    //         //     });
                    //         // }
                    //         3=>{
                    //             is_free_surface.store(x.y().cmpeq(self.p.res[1]-1) & self.v.at_index(x+make_uint3(0,1,0)).cmplt(-thr));
                    //             if_!(is_free_surface.load(), {
                    //                 dv.store(dv.load() + self.unit_mass() * self.v.at_index(x+make_uint3(0,1,0)));
                    //             });
                    //         }
                    //         // 4=>{
                    //         //     is_free_surface.store(x.y().cmpeq(0) & self.v.at_index(x).cmpgt(thr));
                    //         //     if_!(is_free_surface.load(), {
                    //         //         dv.store(dv.load() - self.unit_mass() * self.v.at_index(x));
                    //         //     });
                    //         // }
                    //         // 5=>{
                    //         //     is_free_surface.store(x.z().cmpeq(self.p.res[2]-1) & self.w.at_index(x+make_uint3(0,0,1)).cmplt(-thr));
                    //         //     if_!(is_free_surface.load(), {
                    //         //         dw.store(dw.load() + self.unit_mass() * self.w.at_index(x+make_uint3(0,0,1)));
                    //         //     });
                    //         // }
                    //         // 6=>{
                    //         //     is_free_surface.store(x.z().cmpeq(0) & self.w.at_index(x).cmpgt(thr));
                    //         //     if_!(is_free_surface.load(), {
                    //         //         dw.store(dw.load() - self.unit_mass() * self.w.at_index(x));
                    //         //     });
                    //         // }
                    //         _=>{}
                    //     }
                    //     if_!(is_free_surface.load(), {
                    //         diag.store(diag.load() + self.unit_mass() / 0.5 * lhs_scale);
                    //         // A_coeff.write(i * A.offsets.len() as u32 + offset_idx, 0.0);
                    //     });
                    // }
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
                let weighted_velocity =
                    |u: &Grid<f32>, density_u: &Grid<f32>, mass_u: &Grid<f32>, x: Expr<Uint3>| {
                        let rho = density_u.at_index(x);
                        assert(rho.cmpgt(0.0));
                        let rhs_scale = 1.0 / (h * rho);
                        u.at_index(x) * mass_u.at_index(x) * rhs_scale
                    };
                du.store(
                    du.load()
                        + weighted_velocity(
                            &self.u,
                            &self.density_u,
                            &self.mass_u,
                            x + offset.yxx(),
                        )
                        - weighted_velocity(&self.u, &self.density_u, &self.mass_u, x),
                );
                dv.store(
                    dv.load()
                        + weighted_velocity(
                            &self.v,
                            &self.density_v,
                            &self.mass_v,
                            x + offset.xyx(),
                        )
                        - weighted_velocity(&self.v, &self.density_v, &self.mass_v, x),
                );
                if self.dimension == 3 {
                    dw.store(
                        dw.load()
                            + weighted_velocity(
                                &self.w,
                                &self.density_w,
                                &self.mass_w,
                                x + offset.xxy(),
                            )
                            - weighted_velocity(&self.w, &self.density_w, &self.mass_w, x),
                    );
                }
            }
            let div = if self.dimension == 2 {
                -du.load() - dv.load()
            } else {
                -du.load() - dv.load() - dw.load()
            };
            // cpu_dbg!(make_float3(du.load(), dv.load(), dw.load()));
            self.rhs.set_index(x, div);
        });
    }

    pub fn advect_particle(&self, dt: f32) {
        self.zero_kernel
            .dispatch([self.u.values.len() as u32, 1, 1], &self.u.values)
            .unwrap();
        self.zero_kernel
            .dispatch([self.v.values.len() as u32, 1, 1], &self.v.values)
            .unwrap();
        self.zero_kernel
            .dispatch([self.w.values.len() as u32, 1, 1], &self.w.values)
            .unwrap();
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
        // solver
        //     .zero
        //     .dispatch([self.p.values.len() as u32, 1, 1], &self.p.values)
        //     .unwrap();
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
            panic!("pressure solver failed");
        } else {
            log::info!("pressure solve finished in {} iterations", i.unwrap());
        }

        // pcgsolver::bridson_solve(self.A.as_ref().unwrap(), &self.rhs.values, &self.p.values);
        // pcgsolver::eigen_solve(self.A.as_ref().unwrap(), &self.rhs.values, &self.p.values);

        self.velocity_update_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1, &dt)
            .unwrap();
        // dbg!(&self.v.values.copy_to_vec());
    }
    fn add_particles_to_cell(&self) {
        self.p.reset_particle_list();
        self.build_particle_list_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();
        // {
        //     let mut count = 0;
        //     let list = self.p.cell_particle_list.as_ref().unwrap();
        //     let head = list.head.copy_to_vec();
        //     let next = list.next.copy_to_vec();
        //     for h in &head {
        //         let mut p = *h;
        //         while p != u32::MAX {
        //             count += 1;
        //             p = next[p as usize];
        //         }
        //     }
        //     assert_eq!(count, self.particles_vec.len());
        // }
    }
    fn transfer_particles_to_grid(&self) {
        self.zero_kernel
            .dispatch(
                [self.density_u.values.len() as u32, 1, 1],
                &self.density_u.values,
            )
            .unwrap();
        self.zero_kernel
            .dispatch(
                [self.density_v.values.len() as u32, 1, 1],
                &self.density_v.values,
            )
            .unwrap();
        self.zero_kernel
            .dispatch(
                [self.density_w.values.len() as u32, 1, 1],
                &self.density_w.values,
            )
            .unwrap();
        self.particle_to_grid_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1)
            .unwrap();
        for _ in 0..3 {
            // dbg!(self
            //     .density_u
            //     .values
            //     .copy_to_vec()
            //     .iter()
            //     .filter(|x| **x == 0.0)
            //     .count());
            self.extrapolate_density_kernel
                .as_ref()
                .unwrap()
                .dispatch(self.res_p1)
                .unwrap();
        }
    }
    fn transfer_grid_to_particles(&self) {
        self.grid_to_particle_kernel
            .as_ref()
            .unwrap()
            .dispatch([self.particles_vec.len() as u32, 1, 1])
            .unwrap();

        // dbg!(self.particles.as_ref().unwrap().copy_to_vec());
    }
    fn compute_fluid_phi(&self) {
        self.compute_phi_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.p.res)
            .unwrap();
        // dbg!(self.fluid_phi.values.copy_to_vec());
        if self.log_volume {
            let phi = self.fluid_phi.values.copy_to_vec();
            let non_empty = phi.iter().filter(|&&x| x < 0.0).count();
            log::info!("non empty cells: {}", non_empty);
        }
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
            profile("add_particles_to_cell", || {
                self.add_particles_to_cell();
            });
            profile("transfer_particles_to_grid", || {
                self.transfer_particles_to_grid();
                self.copy_velocity();
            });
            profile("compute_fluid_phi", || {
                self.compute_fluid_phi();
            });
            profile("apply_ext_forces", || {
                self.apply_ext_forces(dt);
                // self.enforce_boundary();
            });
            profile("compute_fluid_mass", || {
                self.compute_fluid_mass();
            });

            profile("solve_pressure", || {
                self.solve_pressure(dt);
            });
            profile("extrapolate_velocity", || {
                self.extrapolate_velocity();
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
    //  theta = cur_phi / (cur_phi - neighbor_phi)
    fn free_surface_theta(&self, cur_phi: Expr<f32>, neighbor_phi: Expr<f32>) -> Expr<f32> {
        assert(cur_phi.cmplt(0.0));
        cur_phi / (cur_phi - neighbor_phi)
    }
    fn enforce_boundary_impl(&self, g: &Grid<f32>, axis: usize) {
        let x = dispatch_id();
        let x_a = x.at(axis);
        if_!(!g.oob(x.int()), {
            let v = g.at_index(x);
            if_!(x_a.cmpeq(0), {
                g.set_index(x, v.max(0.0));
            });
            if_!(x_a.cmpeq(g.res[axis] - 1), {
                g.set_index(x, v.min(0.0));
            });
        });
    }

    // Make sure the fluid does not penetrate the boundary
    fn enforce_boundary(&self) {
        self.enforce_boundary_kernel
            .as_ref()
            .unwrap()
            .dispatch(self.res_p1)
            .unwrap();
    }
    fn extrapolate_velocity(&self) {
        let s = self.device.default_stream();
        s.with_scope(|s| {
            for _ in 0..2 {
                s.submit([
                    self.has_value_u
                        .values
                        .copy_to_buffer_async(&self.tmp_has_value_u.values),
                    self.has_value_v
                        .values
                        .copy_to_buffer_async(&self.tmp_has_value_v.values),
                    self.has_value_w
                        .values
                        .copy_to_buffer_async(&self.tmp_has_value_w.values),
                    self.extrapolate_velocity_kernel
                        .as_ref()
                        .unwrap()
                        .dispatch_async(self.res_p1),
                ])
                .unwrap();
            }
        });
    }
    fn extrapolate_density_impl(&self) {
        let i = dispatch_id();
        let map = |density: &Grid<f32>| {
            if_!(!density.oob(i.int()), {
                let found_values = var!(u32, 0);
                let rho = var!(f32, 0.0);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let x = i.int() + make_int3(dx, dy, dz);
                            if_!(!density.oob(x), {
                                if_!(density.at_index(x.uint()).cmpgt(0.0), {
                                    rho.store(rho.load() + density.at_index(x.uint()));
                                    found_values.store(found_values.load() + 1);
                                });
                            });
                        }
                    }
                }
                let found_values = found_values.load();
                if_!(found_values.cmpgt(0) & density.at_index(i).cmpeq(0.0), {
                    assert(rho.load().cmpgt(0.0));
                    density.set_index(i, rho.load() / found_values.float());
                })
            });
        };
        map(&self.density_u);
        map(&self.density_v);
        map(&self.density_w);
    }
    fn extrapolate_velocity_impl(&self) {
        let i = dispatch_id();
        let use_nearest = false;
        let map = |u: &Grid<f32>, has_value: &Grid<bool>, new_has_value: &Grid<bool>| {
            if_!(!u.oob(i.int()), {
                if_!(!has_value.at_index(i), {
                    let nearest_phi = var!(f32, 1e5);
                    let vel = var!(f32, 0.0);
                    let found_values = var!(u32, 0);
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let x = i.int() + make_int3(dx, dy, dz);
                                if_!(!u.oob(x), {
                                    if use_nearest {
                                        let x_f = u.pos_i_to_f(x);
                                        let x_i = self.fluid_phi.pos_f_to_i(x_f);
                                        if_!(!self.fluid_phi.oob(x_i), {
                                            let x_phi = self.fluid_phi.interpolate(x_f);
                                            if_!(has_value.at_index(x.uint()), {
                                                found_values.store(found_values.load() + 1);
                                                if_!(x_phi.cmplt(nearest_phi.load()), {
                                                    nearest_phi.store(x_phi);
                                                    vel.store(u.at_index(x.uint()));
                                                });
                                            });
                                        });
                                    } else {
                                        // use average
                                        if_!(has_value.at_index(x.uint()), {
                                            found_values.store(found_values.load() + 1);
                                            vel.store(vel.load() + u.at_index(x.uint()));
                                        });
                                    }
                                });
                            }
                        }
                    }
                    if_!(found_values.load().cmpgt(0), {
                        if use_nearest {
                            u.set_index(i, vel.load());
                        } else {
                            u.set_index(i, vel.load() / found_values.load().float());
                        }
                        new_has_value.set_index(i, Bool::from(true));
                    });
                });
            });
        };
        map(&self.u, &self.tmp_has_value_u, &self.has_value_u);
        map(&self.v, &self.tmp_has_value_v, &self.has_value_v);
        if self.dimension == 3 {
            map(&self.w, &self.tmp_has_value_w, &self.has_value_w);
        }
    }
    fn update_velocity_impl(&self, dt: Expr<f32>) {
        // u = u - dt * grad(p)
        let i = dispatch_id();

        let update = |u: &Grid<f32>, density_u: &Grid<f32>, mass_u: &Grid<f32>, axis: u8| {
            let i = i.int();
            if_!(!u.oob(i), {
                let off = match axis {
                    0 => make_int3(1, 0, 0),
                    1 => make_int3(0, 1, 0),
                    2 => make_int3(0, 0, 1),
                    _ => unreachable!(),
                };
                let i_a = i.at(axis as usize);
                let mass = mass_u.at_index(i.uint());
                let rho = density_u.at_index(i.uint());
                if_!(mass.cmpgt(0.0) & i_a.cmpgt(0) & i_a.cmplt(u.res[axis as usize] - 1), {
                    let u_cur = u.at_index(i.uint());
                    // they should be equivalent due to pressure solve
                    let phi_left = self.fluid_phi.at_index((i - off).uint());
                    let phi_right = self.fluid_phi.at_index(i.uint());
                    let both_in_fluid = phi_left.cmplt(0.0) & phi_right.cmplt(0.0);
                    let both_in_air = phi_left.cmpge(0.0) & phi_right.cmpge(0.0);
                    if_!(both_in_fluid, {
                        let grad_pressure =
                            (self.p.at_index(i.uint()) - self.p.at_index((i - off).uint())) / self.h();
                        // let grad_pressure =
                        //     (self.p.at_index((i + off).uint()) - self.p.at_index(i.uint())) / self.h();

                        u.set_index(i.uint(), u_cur - dt * grad_pressure / rho);
                    }, else {
                        if_!(!both_in_air, {
                            if_!(phi_left.cmplt(0.0),{
                                let theta = self.free_surface_theta(phi_left, phi_right);
                                assert(theta.is_finite());
                                assert(theta.cmpge(0.0) & theta.cmple(1.0));
                                let theta = theta.max(1e-2);
                                let grad_pressure = (- self.p.at_index((i - off).uint())  /  theta) / self.h();
                                u.set_index(i.uint(), u_cur - dt * grad_pressure / rho);
                            }, else {
                                let theta = self.free_surface_theta(phi_right, phi_left);
                                assert(theta.is_finite());
                                assert(theta.cmpge(0.0) & theta.cmple(1.0));
                                let theta = theta.max(1e-2);
                                let grad_pressure = (self.p.at_index(i.uint())  /  theta) / self.h();
                                u.set_index(i.uint(), u_cur - dt * grad_pressure / rho);
                            });
                        }, else {
                            u.set_index(i.uint(), u_cur);
                        });
                    });
                    // let grad_pressure =
                    //         (self.p.at_index(i.uint()) - self.p.at_index((i - off).uint())) / self.h();

                    // u.set_index(i.uint(), u_cur - dt * grad_pressure / self.state.rho);
                }, else{
                    // set component to zero
                    u.set_index(i.uint(), 0.0.into());
                });
            });
        };
        update(&self.u, &self.density_u, &self.mass_u, 0);
        update(&self.v, &self.density_v, &self.mass_v, 1);
        if self.dimension == 3 {
            update(&self.w, &self.density_w, &self.mass_w, 2);
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
