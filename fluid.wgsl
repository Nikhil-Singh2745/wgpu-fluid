struct SimParams {
    grid_size: u32,
    mouse_down: u32,
    jacobi_iterations: u32,
    _pad0: u32,
    dt: f32,
    viscosity: f32,
    dissipation: f32,
    add_strength: f32,
    mouse_pos: vec2<f32>,
    mouse_delta: vec2<f32>,
    radius: f32,
    _pad1: vec3<f32>,
}
