use std::ffi::{c_char, c_void};

#[link(name = "cpp_extra")]
extern "C" {
    pub fn create_viewer(particle_count: usize) -> *mut c_void;
    pub fn launch_viewer(viewer: *mut c_void);
    pub fn viewer_set_points(viewer: *mut c_void, points: *const f32, velocity: *const f32);
    pub fn viewer_set_tags(viewer: *mut c_void, tags: *const i32);
    pub fn destroy_viewer(viewer: *mut c_void);
    pub fn viewer_load_mesh(
        viewer: *mut c_void,
        path: *const c_char,
        translate_scale:*const f32,
        nV: *mut i32,
        nF: *mut i32,
    ) -> bool;
    pub fn viewer_mesh_vertices(viewer: *mut c_void) -> *const f64;
    pub fn viewer_mesh_faces(viewer: *mut c_void) -> *const i32;
}
