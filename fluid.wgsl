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
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var velocity: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(2) var velocity_tmp: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(3) var density: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(4) var density_tmp: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(5) var pressure: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(6) var pressure_tmp: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(7) var divergence: texture_storage_2d<rgba16float, read_write>;

fn clamp_coord(p: vec2<f32>, size: f32) -> vec2<i32> {
    return vec2<i32>(clamp(p, vec2<f32>(0.5), vec2<f32>(size - 1.5)));
}

@compute @workgroup_size(8, 8)
fn add_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = f32(params.grid_size);
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    if (params.mouse_down == 0u) { return; }
    let pos = vec2<f32>(f32(gid.x), f32(gid.y));
    let d = distance(pos, params.mouse_pos);
    let falloff = exp(- (d * d) / (params.radius * params.radius + 0.0001));
    let add_dye = params.add_strength * falloff;
    let add_vel = params.mouse_delta * falloff;
    let v = textureLoad(velocity, vec2<i32>(gid.xy)).xy;
    textureStore(velocity, vec2<i32>(gid.xy), vec4<f32>(v + add_vel, 0.0, 0.0));
    let c = textureLoad(density, vec2<i32>(gid.xy));
    textureStore(density, vec2<i32>(gid.xy), vec4<f32>(c.x + add_dye, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_vec(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = f32(params.grid_size);
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    let coord = vec2<f32>(f32(gid.x), f32(gid.y));
    let vel = textureLoad(velocity, vec2<i32>(gid.xy)).xy;
    let prev = coord - vel * params.dt;
    let p = clamp_coord(prev, size);
    let sampled = textureLoad(velocity, p).xy * params.dissipation;
    textureStore(velocity_tmp, vec2<i32>(gid.xy), vec4<f32>(sampled, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_scalar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = f32(params.grid_size);
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    let coord = vec2<f32>(f32(gid.x), f32(gid.y));
    let vel = textureLoad(velocity, vec2<i32>(gid.xy)).xy;
    let prev = coord - vel * params.dt;
    let p = clamp_coord(prev, size);
    let sampled = textureLoad(density, p).x * params.dissipation;
    textureStore(density_tmp, vec2<i32>(gid.xy), vec4<f32>(sampled, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn diffuse_vec(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    let alpha = params.viscosity * params.dt * f32(params.grid_size * params.grid_size);
    let beta = 1.0 + 4.0 * alpha;
    let p = vec2<i32>(gid.xy);
    let center = textureLoad(velocity_tmp, p).xy;
    let l = textureLoad(velocity_tmp, p + vec2<i32>(-1, 0)).xy;
    let r = textureLoad(velocity_tmp, p + vec2<i32>(1, 0)).xy;
    let b = textureLoad(velocity_tmp, p + vec2<i32>(0, -1)).xy;
    let t = textureLoad(velocity_tmp, p + vec2<i32>(0, 1)).xy;
    let out_v = (center + alpha * (l + r + b + t)) / beta;
    textureStore(velocity, p, vec4<f32>(out_v, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn compute_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(velocity, p + vec2<i32>(-1, 0)).xy;
    let r = textureLoad(velocity, p + vec2<i32>(1, 0)).xy;
    let b = textureLoad(velocity, p + vec2<i32>(0, -1)).xy;
    let t = textureLoad(velocity, p + vec2<i32>(0, 1)).xy;
    let h = 1.0;
    let div = 0.5 * ((r.x - l.x) / h + (t.y - b.y) / h);
    textureStore(divergence, p, vec4<f32>(div, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn pressure_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(pressure, p + vec2<i32>(-1, 0)).x;
    let r = textureLoad(pressure, p + vec2<i32>(1, 0)).x;
    let b = textureLoad(pressure, p + vec2<i32>(0, -1)).x;
    let t = textureLoad(pressure, p + vec2<i32>(0, 1)).x;
    let div = textureLoad(divergence, p).x;
    let out_p = (l + r + b + t - div) * 0.25;
    textureStore(pressure_tmp, p, vec4<f32>(out_p, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn subtract_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.grid_size || gid.y >= params.grid_size) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(pressure, p + vec2<i32>(-1, 0)).x;
    let r = textureLoad(pressure, p + vec2<i32>(1, 0)).x;
    let b = textureLoad(pressure, p + vec2<i32>(0, -1)).x;
    let t = textureLoad(pressure, p + vec2<i32>(0, 1)).x;
    let vel = textureLoad(velocity, p).xy;
    let grad = vec2<f32>(r - l, t - b) * 0.5;
    textureStore(velocity, p, vec4<f32>(vel - grad, 0.0, 0.0));
}

@group(0) @binding(0) var density_tex: texture_2d<f32>;
@group(0) @binding(1) var density_sampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );
    var out: VSOut;
    out.pos = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = uv[vid];
    return out;
}

@fragment
fn fs_draw(in: VSOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(density_tex, density_sampler, in.uv, 0.0).x;
    let glow = vec3<f32>(0.2, 0.8, 1.2) * c;
    return vec4<f32>(glow, 1.0);
}