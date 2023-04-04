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
use luisa::init_logger;
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
            rho: 1.0,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
        },
    );
    for z in 0..50 {
        for y in 0..80 {
            for x in 0..100 {
                let x = x as f32 * 0.02;
                let y = y as f32 * 0.02;
                let z = z as f32 * 0.02 + 0.4;
                sim.particles_vec.push(Particle {
                    pos: Float3::new(x, y, z),
                    vel: Float3::new(0.0, 0.0, 0.0),
                    radius: h / (3.0f32).sqrt(),
                    c_x: Float3::new(0.0, 0.0, 0.0),
                    c_y: Float3::new(0.0, 0.0, 0.0),
                    c_z: Float3::new(0.0, 0.0, 0.0),
                })
            }
        }
    }
    sim.commit();
    launch_viewer_for_sim(&mut sim);
}
fn single_particle(device: Device, res: u32, dt: f32) {
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
            rho: 10.0,
            dimension: 3,
            transfer: ParticleTransfer::Apic,
            advect: VelocityIntegration::Euler,
            preconditioner: Preconditioner::DiagJacobi,
        },
    );
    sim.particles_vec.push(Particle {
        pos: Float3::new(0.5, 1.0, 1.0),
        vel: Float3::new(0.0, 0.0, 0.0),
        radius: h * (3.0f32).sqrt(),
        c_x: Float3::new(0.0, 0.0, 0.0),
        c_y: Float3::new(0.0, 0.0, 0.0),
        c_z: Float3::new(0.0, 0.0, 0.0),
    });
    sim.commit();
    launch_viewer_for_sim(&mut sim);
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
        // let mut input = String::new();
        // match std::io::stdin().read_line(&mut input) {
        //     Ok(_goes_into_input_above) => {},
        //     Err(_no_updates_is_fine) => {},
        // }
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
    dambreak(device, 40, 1.0 / 30.0);
    // single_particle(device, 32, 1.0/30.0);
    // stability(device, 32, 1.0/30.0);
}
