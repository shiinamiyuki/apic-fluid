use std::{env::args, ffi::c_void, io::Read, time::Duration};

use apic_fluid::{fluid::Replay, *};
use luisa::init_logger;

fn main() {
    init_logger();
    let scene = args().nth(1).unwrap();
    let file = format!("replays/{}", scene);
    let replay = Replay::load(&file);
    launch_viewer(&replay);
}

fn launch_viewer(replay: &Replay) {
    let count = replay.frames[0].len();
    let viewer = unsafe { cpp_extra::create_viewer(count) };

    let viewer_thread = {
        let viewer = viewer as u64;
        std::thread::spawn(move || unsafe {
            let viewer = viewer as *mut c_void;
            cpp_extra::launch_viewer(viewer);
        })
    };
    let buf = &replay.frames[0];
    let tags = buf.iter().map(|p| p.tag as i32).collect::<Vec<_>>();
    let mut particle_pos = vec![0.0f32; count * 3];
    let mut particle_vel = vec![0.0f32; count * 3];
    unsafe { cpp_extra::viewer_set_tags(viewer, tags.as_ptr()) }
    let mut frame = 0;
    while !viewer_thread.is_finished() {
        let buf = &replay.frames[frame % replay.frames.len()];
        unsafe {
            for i in 0..count {
                particle_pos[3 * i + 0] = buf[i].pos[0];
                particle_pos[3 * i + 1] = buf[i].pos[1];
                particle_pos[3 * i + 2] = buf[i].pos[2];

                particle_vel[3 * i + 0] = buf[i].vel[0];
                particle_vel[3 * i + 1] = buf[i].vel[1];
                particle_vel[3 * i + 2] = buf[i].vel[2];
            }
            cpp_extra::viewer_set_points(viewer, particle_pos.as_ptr(), particle_vel.as_ptr());
        }
        frame += 1;
        std::thread::sleep(Duration::from_secs_f64(replay.config.dt as f64));
    }

    unsafe {
        cpp_extra::destroy_viewer(viewer);
    }
}
