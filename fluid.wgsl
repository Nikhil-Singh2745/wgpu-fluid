struct SimParams {
    grid_size: u32,
    mouse_down: u32,
    dt: f32,
    viscosity: f32,
    dissipation: f32,
    add_strength: f32,
    mouse_pos: vec2<f32>,
    mouse_delta: vec2<f32>,
    radius: f32,
    _pad: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var velocity: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(2) var velocity_tmp: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(3) var density: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(4) var density_tmp: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(5) var pressure: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(6) var pressure_tmp: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(7) var divergence_tex: texture_storage_2d<rgba16float, read_write>;

fn in_bounds(gid: vec3<u32>) -> bool {
    return gid.x < params.grid_size && gid.y < params.grid_size;
}

fn safe_load_vel(p: vec2<i32>) -> vec2<f32> {
    let size = i32(params.grid_size);
    let cp = clamp(p, vec2<i32>(0), vec2<i32>(size - 1));
    return textureLoad(velocity, cp).xy;
}

fn safe_load_vel_tmp(p: vec2<i32>) -> vec2<f32> {
    let size = i32(params.grid_size);
    let cp = clamp(p, vec2<i32>(0), vec2<i32>(size - 1));
    return textureLoad(velocity_tmp, cp).xy;
}

fn safe_load_dens(p: vec2<i32>) -> f32 {
    let size = i32(params.grid_size);
    let cp = clamp(p, vec2<i32>(0), vec2<i32>(size - 1));
    return textureLoad(density, cp).x;
}

fn safe_load_press(p: vec2<i32>) -> f32 {
    let size = i32(params.grid_size);
    let cp = clamp(p, vec2<i32>(0), vec2<i32>(size - 1));
    return textureLoad(pressure, cp).x;
}

fn safe_load_press_tmp(p: vec2<i32>) -> f32 {
    let size = i32(params.grid_size);
    let cp = clamp(p, vec2<i32>(0), vec2<i32>(size - 1));
    return textureLoad(pressure_tmp, cp).x;
}

fn safe_load_div(p: vec2<i32>) -> f32 {
    let size = i32(params.grid_size);
    let cp = clamp(p, vec2<i32>(0), vec2<i32>(size - 1));
    return textureLoad(divergence_tex, cp).x;
}

@compute @workgroup_size(8, 8)
fn add_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    if (params.mouse_down == 0u) { return; }
    
    let pos = vec2<f32>(f32(gid.x), f32(gid.y));
    let d = distance(pos, params.mouse_pos);
    let falloff = exp(-(d * d) / (params.radius * params.radius + 0.001));
    
    let p = vec2<i32>(gid.xy);
    let v = textureLoad(velocity, p).xy;
    let add_vel = params.mouse_delta * falloff * 50.0;
    textureStore(velocity, p, vec4<f32>(v + add_vel, 0.0, 0.0));
    
    let c = textureLoad(density, p).x;
    let add_dye = params.add_strength * falloff;
    textureStore(density, p, vec4<f32>(c + add_dye, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_vel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    
    let p = vec2<i32>(gid.xy);
    let pos = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let vel = textureLoad(velocity, p).xy;
    let prev_pos = pos - vel * params.dt;
    
    let size = f32(params.grid_size);
    let pp = clamp(prev_pos - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(size - 1.001));
    let i = vec2<i32>(floor(pp));
    let f = fract(pp);
    
    let v00 = safe_load_vel(i);
    let v10 = safe_load_vel(i + vec2<i32>(1, 0));
    let v01 = safe_load_vel(i + vec2<i32>(0, 1));
    let v11 = safe_load_vel(i + vec2<i32>(1, 1));
    
    let v0 = mix(v00, v10, f.x);
    let v1 = mix(v01, v11, f.x);
    let sampled = mix(v0, v1, f.y) * params.dissipation;
    
    textureStore(velocity_tmp, p, vec4<f32>(sampled, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn copy_vel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let v = textureLoad(velocity_tmp, p).xy;
    textureStore(velocity, p, vec4<f32>(v, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_dens(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    
    let p = vec2<i32>(gid.xy);
    let pos = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let vel = textureLoad(velocity, p).xy;
    let prev_pos = pos - vel * params.dt;
    
    let size = f32(params.grid_size);
    let pp = clamp(prev_pos - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(size - 1.001));
    let i = vec2<i32>(floor(pp));
    let f = fract(pp);
    
    let d00 = safe_load_dens(i);
    let d10 = safe_load_dens(i + vec2<i32>(1, 0));
    let d01 = safe_load_dens(i + vec2<i32>(0, 1));
    let d11 = safe_load_dens(i + vec2<i32>(1, 1));
    
    let d0 = mix(d00, d10, f.x);
    let d1 = mix(d01, d11, f.x);
    let sampled = mix(d0, d1, f.y) * params.dissipation;
    
    textureStore(density_tmp, p, vec4<f32>(sampled, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn copy_dens(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let d = textureLoad(density_tmp, p).x;
    textureStore(density, p, vec4<f32>(d, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn compute_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    
    let vL = safe_load_vel(p + vec2<i32>(-1, 0)).x;
    let vR = safe_load_vel(p + vec2<i32>(1, 0)).x;
    let vB = safe_load_vel(p + vec2<i32>(0, -1)).y;
    let vT = safe_load_vel(p + vec2<i32>(0, 1)).y;
    
    let div = 0.5 * (vR - vL + vT - vB);
    textureStore(divergence_tex, p, vec4<f32>(div, 0.0, 0.0, 0.0));
    
    textureStore(pressure, p, vec4<f32>(0.0));
    textureStore(pressure_tmp, p, vec4<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn pressure_jacobi_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    
    let pL = safe_load_press(p + vec2<i32>(-1, 0));
    let pR = safe_load_press(p + vec2<i32>(1, 0));
    let pB = safe_load_press(p + vec2<i32>(0, -1));
    let pT = safe_load_press(p + vec2<i32>(0, 1));
    let div = safe_load_div(p);
    
    let new_p = (pL + pR + pB + pT - div) * 0.25;
    textureStore(pressure_tmp, p, vec4<f32>(new_p, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn pressure_jacobi_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    
    let pL = safe_load_press_tmp(p + vec2<i32>(-1, 0));
    let pR = safe_load_press_tmp(p + vec2<i32>(1, 0));
    let pB = safe_load_press_tmp(p + vec2<i32>(0, -1));
    let pT = safe_load_press_tmp(p + vec2<i32>(0, 1));
    let div = safe_load_div(p);
    
    let new_p = (pL + pR + pB + pT - div) * 0.25;
    textureStore(pressure, p, vec4<f32>(new_p, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn subtract_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    
    let pL = safe_load_press(p + vec2<i32>(-1, 0));
    let pR = safe_load_press(p + vec2<i32>(1, 0));
    let pB = safe_load_press(p + vec2<i32>(0, -1));
    let pT = safe_load_press(p + vec2<i32>(0, 1));
    
    let grad = vec2<f32>(pR - pL, pT - pB) * 0.5;
    let vel = textureLoad(velocity, p).xy;
    textureStore(velocity, p, vec4<f32>(vel - grad, 0.0, 0.0));
}

@group(1) @binding(0) var render_density_tex: texture_2d<f32>;
@group(1) @binding(1) var render_sampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );
    var out: VSOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_draw(in: VSOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(render_density_tex, render_sampler, in.uv, 0.0).x;
    let intensity = clamp(c, 0.0, 1.0);
    let color = vec3<f32>(0.1, 0.4, 0.9) * intensity + vec3<f32>(0.0, 0.7, 1.0) * intensity * intensity;
    return vec4<f32>(color, 1.0);
}