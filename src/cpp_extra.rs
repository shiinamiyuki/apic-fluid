use std::ffi::c_void;

#[link(name = "cpp_extra")]
extern "C" {
    pub fn launch_viewer(particle_count: usize) -> *mut c_void;
    pub fn viewer_set_points(viewer: *mut c_void, points: *const f32);
    pub fn destroy_viewer(viewer: *mut c_void);
}
