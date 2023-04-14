use std::env::{args, current_exe};

use apic_fluid::*;

use apic_fluid::fluid::*;
use apic_fluid::reconstruction::AnisotropicDiffusion;
use luisa::init_logger;

fn main() {
    init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_cpu_device().unwrap();
    let scene = args().nth(1).unwrap();
    let file = format!("replays/{}", scene);
    let replay = Replay::load(&file);
    let res = args().nth(2).unwrap().parse::<u32>().unwrap();
    let r = args().nth(3).unwrap().parse::<f32>().unwrap();
    let frame_start = args()
        .nth(4)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    let frame_end = args()
        .nth(5)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(replay.frames.len());
    let frame_step = args()
        .nth(6)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(1);
    let buf = device
        .create_buffer::<Particle>(replay.frames[0].len())
        .unwrap();
    let min_res = replay.config.res.iter().copied().min().unwrap();
    let fres = [
        replay.config.res[0] as f32,
        replay.config.res[1] as f32,
        replay.config.res[2] as f32,
    ];
    let extent = [
        fres[0] * replay.config.h,
        fres[1] * replay.config.h,
        fres[2] * replay.config.h,
    ];
    let min_extent = min_res as f32 * replay.config.h;
    let recon_h = min_extent / res as f32;
    let frecon_res = [
        extent[0] / recon_h,
        extent[1] / recon_h,
        extent[2] / recon_h,
    ];
    let recon_res = [
        frecon_res[0].ceil() as u32,
        frecon_res[1].ceil() as u32,
        frecon_res[2].ceil() as u32,
    ];
    let reconstruction = AnisotropicDiffusion::new(
        device.clone(),
        recon_res,
        recon_h,
        r,
        replay.frames[0].len(),
    );
    let mut frame = frame_start;
    while frame < frame_end {
        buf.fill_fn(|i| {
            let p = replay.frames[frame][i];
            Particle {
                pos: Float3::new(p.pos[0], p.pos[1], p.pos[2]),
                vel: Float3::new(p.vel[0], p.vel[1], p.vel[2]),
                tag: p.tag,
                ..Default::default()
            }
        });
        reconstruction.save_obj(
            &buf,
            frame,
            &format!("output_meshes/{}", scene),
        );
        frame += frame_step;
    }
}
