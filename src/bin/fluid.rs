use std::env::current_exe;

use apic_fluid::{fluid::*, pcgsolver::Preconditioner, *};
use luisa::init_logger;
use rand::{rngs::StdRng, *};

fn test_solve() {
    init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_cpu_device().unwrap();
    let mut sim = Simulation::new(
        device.clone(),
        SimulationSettings {
            dt: 1.0,
            max_iterations: 1024,
            tolerance: 1e-4,
            res: [512, 512, 1],
            h: 1.0,
            rho:1.0,
            dimension: 2,
            transfer: ParticleTransfer::Pic,
            advect: VelocityIntegration::RK3,
            preconditioner: Preconditioner::IncompletePoisson,
        },
    );
    sim.particles_vec = vec![Particle::default(); 100];
    let mut rng = StdRng::seed_from_u64(0);
    sim.u.values.fill_fn(|_| rng.gen::<f32>() * 2.0 - 1.0);
    sim.v.values.fill_fn(|_| rng.gen::<f32>() * 2.0 - 1.0);
    sim.commit();
    println!("committed");
    sim.solve_pressure(sim.settings.dt);
}
fn main() {
    test_solve();
}
