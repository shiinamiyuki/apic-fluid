use std::{
    env::current_exe,
    ffi::c_void,
    sync::{atomic::AtomicBool, Arc},
};

use apic_fluid::{
    fluid::*,
    pcgsolver::{eigen_solve, PcgSolver, Preconditioner, Stencil},
    *,
};
use luisa::{glam::Vec3, init_logger};
use parking_lot::Mutex;
use rand::{rngs::StdRng, *};

fn test_solve() {
    init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_cpu_device().unwrap();
    let n = 100;
    let offsets = [
        Int3::new(0, 0, 0),
        Int3::new(1, 0, 0),
        Int3::new(-1, 0, 0),
        Int3::new(0, 1, 0),
        Int3::new(0, -1, 0),
        Int3::new(0, 0, 1),
        Int3::new(0, 0, -1),
    ];
    let stencil = Stencil::new(device.clone(), [n, n, n], &offsets);
    let solver = PcgSolver::new(
        device.clone(),
        [n, n, n],
        512,
        1e-5,
        Preconditioner::IncompletePoisson,
    );
    let mut buf = vec![];
    for i in 0..(n * n * n) {
        buf.push(6.0);
        for _ in 0..6 {
            buf.push(-1.0);
        }
    }
    stencil.coeff.copy_from(&buf);
    let mut rng = StdRng::seed_from_u64(0);
    let b = device
        .create_buffer_from_fn((n * n * n) as usize, |_| (rng.gen::<f32>()) * 4.0 - 2.0)
        .unwrap();
    let x = device
        .create_buffer_from_fn((n * n * n) as usize, |_| 0.0)
        .unwrap();
    dbg!(solver.solve(&stencil, &b, &x));
    let my_solution = x.copy_to_vec();
    x.fill(0.0);
    eigen_solve(&stencil, &b, &x);
    let eigen_solution = x.copy_to_vec();
    println!("{:?}", &my_solution[..16]);
    println!("{:?}", &eigen_solution[..16]);
}
fn dambreak(device: Device, res: u32, dt: f32) {
    let extent = 2.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 1024,
            tolerance: 1e-5,
            res: [res, res, res * 2],
            h,
            g: 0.5,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: true,
            seperation_threshold: 0.0,
        },
    );
    let mut rng = StdRng::seed_from_u64(0);
    for z in 0..60 {
        for y in 0..60 {
            for x in 0..100 {
                let x = x as f32 * 0.02;
                let y = y as f32 * 0.02;
                let z = z as f32 * 0.02;

                let x = x + (rng.gen::<f32>() - 0.5) * 0.01;
                let y = y + (rng.gen::<f32>() - 0.5) * 0.01;
                let z = z + (rng.gen::<f32>() - 0.5) * 0.01;
                sim.particles_vec.push(Particle {
                    pos: Float3::new(x, y, z),
                    vel: Float3::new(0.0, 0.0, 0.0),
                    radius: h * 0.5 * (3.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 0,
                    density: 1.0,
                })
            }
        }
    }
    sim.commit();
    launch_viewer_for_sim(&mut sim);
}
fn wave(device: Device, res: u32, dt: f32) {
    let extent = 2.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 1024,
            tolerance: 1e-5,
            res: [res, res, res * 2],
            h,
            g: 0.5,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: false,
            seperation_threshold: 0.1,
        },
    );
    for z in 0..200 {
        for y in 0..10 {
            for x in 0..100 {
                let x = x as f32 * 0.02;
                let y = y as f32 * 0.02;
                let z = z as f32 * 0.02;
                let vel = if z < 0.4 { 1.0 } else { 0.0 };
                sim.particles_vec.push(Particle {
                    pos: Float3::new(x, y, z),
                    vel: Float3::new(0.0, 0.0, vel),
                    radius: h * 0.5 * (3.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 0,
                    density: 1.0,
                })
            }
        }
    }
    sim.commit();
    launch_viewer_for_sim(&mut sim);
}
fn boundary(device: Device, res: u32, dt: f32) {
    let extent = 1.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 1024,
            tolerance: 1e-5,
            res: [res, res, res],
            h,
            g: 0.5,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: true,
            seperation_threshold: 0.1,
        },
    );
    for z in 0..200 {
        for y in 0..1 {
            for x in 0..200 {
                let x = x as f32 * 0.005;
                let y = y as f32 * 0.005 + 1.0;
                let z = z as f32 * 0.005;
                sim.particles_vec.push(Particle {
                    pos: Float3::new(x, y, z),
                    vel: Float3::new(0.0, 0.0, 0.0),
                    radius: h * 0.5 * (3.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 0,
                    density: 1.0,
                })
            }
        }
    }
    sim.commit();
    launch_viewer_for_sim(&mut sim);
}
fn splash(device: Device, res: u32, dt: f32) {
    let extent = 1.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 1024,
            tolerance: 1e-5,
            res: [res * 2, res * 2, res * 2],
            h,
            g: 9.8,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: false,
            seperation_threshold: 0.0,
        },
    );
    for z in 0..10 {
        for y in 0..40 {
            for x in 0..10 {
                let x = x as f32 * 0.02 + 0.9;
                let y = y as f32 * 0.02 + 0.5;
                let z = z as f32 * 0.02 + 0.9;
                sim.particles_vec.push(Particle {
                    pos: Float3::new(x, y, z),
                    vel: Float3::new(0.0, -5.0, 0.0),
                    radius: h * 0.5 * (3.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 0,
                    density: 1.0,
                })
            }
        }
    }

    for z in 0..100 {
        for y in 0..20 {
            for x in 0..100 {
                let x = x as f32 * 0.02;
                let y = y as f32 * 0.02;
                let z = z as f32 * 0.02;
                sim.particles_vec.push(Particle {
                    pos: Float3::new(x, y, z),
                    vel: Float3::new(0.0, 0.0, 0.0),
                    radius: h * 0.5 * (3.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 0,
                    density: 1.0,
                })
            }
        }
    }
    sim.commit();
    launch_viewer_for_sim(&mut sim);
}
fn vortex(device: Device, res: u32, dt: f32) {
    let extent = 1.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 1024,
            tolerance: 1e-5,
            res: [res, res, res],
            h,
            g: 0.0,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: false,
            seperation_threshold: 0.0,
        },
    );
    sim.log_volume = false;
    let mut color_particles = vec![];
    let mut others = vec![];
    let center1 = Vec3::new(0.5, 0.5, 0.2);
    let center2 = Vec3::new(0.5, 0.5, 0.8);
    let radius = 0.1;
    let map = |p, c| {
        let mut d: Vec3 = p - c;
        d.z *= 6.0;
        d.length() < radius
    };
    let n = 300;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let x = x as f32 / n as f32;
                let y = y as f32 / n as f32;
                let z = z as f32 / n as f32;
                let p = Vec3::new(x, y, z);
                if map(p, center1) {
                    color_particles.push(Particle {
                        pos: Float3::new(x, y, z),
                        vel: Float3::new(0.0, 0.0, 4.0),
                        radius: h * 0.5 * (3.0f32).sqrt(),
                        c_x: Float3::new(0.0, 0.0, 0.0),
                        c_y: Float3::new(0.0, 0.0, 0.0),
                        c_z: Float3::new(0.0, 0.0, 0.0),
                        tag: 1,
                        density: 1.0,
                    });
                } else if map(p, center2) {
                    color_particles.push(Particle {
                        pos: Float3::new(x, y, z),
                        vel: Float3::new(0.0, 0.0, -4.0),
                        radius: h * 0.5 * (3.0f32).sqrt(),
                        c_x: Float3::new(0.0, 0.0, 0.0),
                        c_y: Float3::new(0.0, 0.0, 0.0),
                        c_z: Float3::new(0.0, 0.0, 0.0),
                        tag: 2,
                        density: 1.0,
                    });
                } else {
                    others.push(Particle {
                        pos: Float3::new(x, y, z),
                        vel: Float3::new(0.0, 0.0, 0.0),
                        radius: h * 0.5 * (3.0f32).sqrt(),
                        c_x: Float3::new(0.0, 0.0, 0.0),
                        c_y: Float3::new(0.0, 0.0, 0.0),
                        c_z: Float3::new(0.0, 0.0, 0.0),
                        tag: 0,
                        density: 1.0,
                    });
                }
            }
        }
    }
    for p in &color_particles {
        sim.particles_vec.push(*p);
    }
    for p in &others {
        sim.particles_vec.push(*p);
    }
    sim.commit();
    launch_viewer_for_sim_with_tags(&mut sim, color_particles.len());
}
fn vortex_sheet(device: Device, res: u32, dt: f32) {
    let extent = 1.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 1024,
            tolerance: 1e-5,
            res: [res, res, 1],
            h,
            g: 0.0,
            dimension: 2,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: false,
            seperation_threshold: 0.0,
        },
    );
    sim.log_volume = false;
    let mut color_particles = vec![];
    let others = vec![];
    let center = Vec3::new(0.5, 0.5, 0.0);
    let map = |p, c| {
        let d: Vec3 = p - c;
        d.x * d.y >= 0.0
    };
    let n = 500;

    for y in 0..n {
        for x in 0..n {
            let x = x as f32 / n as f32;
            let y = y as f32 / n as f32;
            let p = Vec3::new(x, y, 0.0);
            let r = p - Vec3::new(0.5, 0.5, 0.0);
            let l = Vec3::new(0.0, 0.0, -1.0);
            let mut v = r.cross(l);
            if r.length() > 0.25 {
                v *= 0.0;
            }
            if map(p, center) {
                color_particles.push(Particle {
                    pos: Float3::new(x, y, 0.0),
                    vel: v.into(),
                    radius: h * 0.5 * (2.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 1,
                    density: 1.0,
                });
            } else {
                color_particles.push(Particle {
                    pos: Float3::new(x, y, 0.0),
                    vel: v.into(),
                    radius: h * 0.5 * (2.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 2,
                    density: 1.0,
                });
            }
        }
    }

    for p in &color_particles {
        sim.particles_vec.push(*p);
    }
    for p in &others {
        sim.particles_vec.push(*p);
    }
    sim.commit();
    launch_viewer_for_sim_with_tags(&mut sim, color_particles.len());
}
fn mixed_density(device: Device, res: u32, dt: f32) {
    let extent = 1.0;
    let h = extent / res as f32;
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt,
            max_iterations: 4096,
            tolerance: 1e-5,
            res: [res, res, 1],
            h,
            g: 1.0,
            dimension: 2,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
            force_wall_separation: false,
            seperation_threshold: 0.0,
        },
    );
    let mut color_particles = vec![];
    let others = vec![];
    let n = 500;

    for y in 0..n {
        for x in 0..n {
            let x = x as f32 / n as f32;
            let y = y as f32 / n as f32;

            if y < 0.5 {
                color_particles.push(Particle {
                    pos: Float3::new(x, y, 0.0),
                    vel: Float3::new(0.0, 0.0, 0.0),
                    radius: h * 0.5 * (2.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 1,
                    density: 0.1,
                });
            } else {
                color_particles.push(Particle {
                    pos: Float3::new(x, y, 0.0),
                    vel: Float3::new(0.0, 0.0, 0.0),
                    radius: h * 0.5 * (2.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                    tag: 2,
                    density: 1.0,
                });
            }
        }
    }

    for p in &color_particles {
        sim.particles_vec.push(*p);
    }
    for p in &others {
        sim.particles_vec.push(*p);
    }
    sim.commit();
    launch_viewer_for_sim_with_tags(&mut sim, color_particles.len());
}
fn launch_viewer_for_sim_with_tags(sim: &mut Simulation, count: usize) {
    let viewer = unsafe { cpp_extra::create_viewer(count) };

    let viewer_thread = {
        let viewer = viewer as u64;
        std::thread::spawn(move || unsafe {
            let viewer = viewer as *mut c_void;
            cpp_extra::launch_viewer(viewer);
        })
    };
    let mut buf = sim.particles_vec.clone();
    let tags = buf[..count]
        .iter()
        .map(|p| p.tag as i32)
        .collect::<Vec<_>>();
    let mut particle_pos = vec![0.0f32; count * 3];
    let mut particle_vel = vec![0.0f32; count * 3];
    unsafe { cpp_extra::viewer_set_tags(viewer, tags.as_ptr()) }
    while !viewer_thread.is_finished() {
        unsafe {
            sim.particles.as_ref().unwrap().copy_to(&mut buf);
            for i in 0..count {
                particle_pos[3 * i + 0] = buf[i].pos.x;
                particle_pos[3 * i + 1] = buf[i].pos.y;
                particle_pos[3 * i + 2] = buf[i].pos.z;

                particle_vel[3 * i + 0] = buf[i].vel.x;
                particle_vel[3 * i + 1] = buf[i].vel.y;
                particle_vel[3 * i + 2] = buf[i].vel.z;
            }
            cpp_extra::viewer_set_points(viewer, particle_pos.as_ptr(), particle_vel.as_ptr());
        }
        sim.step();
    }
    unsafe {
        cpp_extra::destroy_viewer(viewer);
    }
}

fn launch_viewer_for_sim(sim: &mut Simulation) {
    let viewer = unsafe { cpp_extra::create_viewer(sim.particles_vec.len()) };

    let viewer_thread = {
        let viewer = viewer as u64;
        std::thread::spawn(move || unsafe {
            let viewer = viewer as *mut c_void;
            cpp_extra::launch_viewer(viewer);
        })
    };
    let mut buf = sim.particles_vec.clone();
    let mut particle_pos = vec![0.0f32; buf.len() * 3];
    let mut particle_vel = vec![0.0f32; buf.len() * 3];
    while !viewer_thread.is_finished() {
        unsafe {
            sim.particles.as_ref().unwrap().copy_to(&mut buf);
            for i in 0..buf.len() {
                particle_pos[3 * i + 0] = buf[i].pos.x;
                particle_pos[3 * i + 1] = buf[i].pos.y;
                particle_pos[3 * i + 2] = buf[i].pos.z;

                particle_vel[3 * i + 0] = buf[i].vel.x;
                particle_vel[3 * i + 1] = buf[i].vel.y;
                particle_vel[3 * i + 2] = buf[i].vel.z;
            }
            cpp_extra::viewer_set_points(viewer, particle_pos.as_ptr(), particle_vel.as_ptr());
        }
        sim.step();
    }
    unsafe {
        cpp_extra::destroy_viewer(viewer);
    }
}
fn main() {
    // test_solve();
    init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_cpu_device().unwrap();
    vortex_sheet(device, 128, 0.01);
    // mixed_density(device, 128, 0.01);
    // dambreak(device, 32, 1.0 / 30.0);
    // wave(device, 64, 1.0 / 30.0);
    // splash(device, 40, 1.0 / 30.0);
    // boundary(device, 64, 1.0/30.0);
    // stability(device, 32, 1.0/30.0);
}
