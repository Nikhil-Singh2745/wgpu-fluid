use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    dpi::LogicalSize,
    event::*,
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    grid_size: u32,
    mouse_down: u32,
    dt: f32,
    viscosity: f32,
    dissipation: f32,
    add_strength: f32,
    mouse_pos: [f32; 2],
    mouse_delta: [f32; 2],
    radius: f32,
    _pad0: f32,
    _pad1: [f32; 4],
}

const GRID_SIZE: u32 = 256;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = bits & 0x007FFFFF;
    if exp <= 0 { 0u16 }
    else if exp >= 31 { (sign | 0x7C00) as u16 }
    else { (sign | ((exp as u32) << 10) | (frac >> 13)) as u16 }
}

fn create_storage_tex(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn main() {
    env_logger::init();

    // WSL2/WSLg has flaky Wayland. Force X11 by clearing WAYLAND_DISPLAY.
    // Must happen BEFORE EventLoop::new().
    std::env::set_var("WAYLAND_DISPLAY", "");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("WGPU Fluid Simulation")
            .with_inner_size(LogicalSize::new(800.0, 800.0))
            .build(&event_loop)
            .unwrap(),
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    }))
    .expect("No suitable GPU adapter found");

    eprintln!("GPU: {}", adapter.get_info().name);
    eprintln!("Backend: {:?}", adapter.get_info().backend);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            required_limits: wgpu::Limits {
                max_storage_textures_per_shader_stage: 8,
                ..wgpu::Limits::default()
            },
        },
        None,
    ))
    .expect("Failed to create device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
    let win_size = window.inner_size();

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: win_size.width.max(1),
        height: win_size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // ---- Shaders ----
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../fluid.wgsl").into()),
    });

    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render_shader"),
        source: wgpu::ShaderSource::Wgsl(r#"
@group(0) @binding(0) var render_density_tex: texture_2d<f32>;
@group(0) @binding(1) var render_sampler: sampler;
@group(0) @binding(2) var render_velocity_tex: texture_2d<f32>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0), vec2<f32>(2.0, 1.0), vec2<f32>(0.0, -1.0)
    );
    var out: VSOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

// HSV to RGB conversion
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if (hp < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hp < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hp < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (hp < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (hp < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(m, m, m);
}

@fragment
fn fs_draw(in: VSOut) -> @location(0) vec4<f32> {
    let dens = textureSampleLevel(render_density_tex, render_sampler, in.uv, 0.0).x;
    let vel = textureSampleLevel(render_velocity_tex, render_sampler, in.uv, 0.0).xy;

    // Velocity magnitude and direction
    let speed = length(vel);
    let angle = atan2(vel.y, vel.x); // -pi to pi

    // Map velocity direction to hue (0..1), speed controls saturation
    let hue = fract(angle / 6.2832 + 0.5);  // normalize -pi..pi to 0..1
    let sat = clamp(speed * 3.0, 0.2, 1.0);  // more speed = more saturated
    let intensity = clamp(dens, 0.0, 1.0);

    // Base color from velocity direction
    let base_color = hsv2rgb(hue, sat, 1.0);

    // Glow: boost bright areas with a power curve
    let glow = pow(intensity, 0.6);        // softer falloff for thin wisps
    let bloom = pow(intensity, 3.0) * 0.8; // hot-white core on dense areas

    // Subtle dark background gradient (not pure black)
    let bg = vec3<f32>(0.01, 0.01, 0.03);

    // Composite: colored fluid + white bloom on top
    let fluid = base_color * glow;
    let white_bloom = vec3<f32>(bloom, bloom, bloom);
    let color = bg * (1.0 - intensity) + fluid + white_bloom;

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#.into()),
    });

    // ---- Textures ----
    let wg = ((GRID_SIZE + 7) / 8, (GRID_SIZE + 7) / 8);

    let (_vel, vel_view) = create_storage_tex(&device, GRID_SIZE);
    let (_vel_tmp, vel_tmp_view) = create_storage_tex(&device, GRID_SIZE);
    let (dens_tex, dens_view) = create_storage_tex(&device, GRID_SIZE);
    let (_dens_tmp, dens_tmp_view) = create_storage_tex(&device, GRID_SIZE);
    let (_press, press_view) = create_storage_tex(&device, GRID_SIZE);
    let (_press_tmp, press_tmp_view) = create_storage_tex(&device, GRID_SIZE);
    let (_div, div_view) = create_storage_tex(&device, GRID_SIZE);

    // Seed density blob
    {
        let g = GRID_SIZE;
        let mut data = vec![[0u16; 4]; (g * g) as usize];
        let (cx, cy, r) = (g as f32 / 2.0, g as f32 / 2.0, 30.0f32);
        for y in 0..g {
            for x in 0..g {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let val = (1.0 - (dx * dx + dy * dy) / (r * r)).max(0.0);
                data[(y * g + x) as usize][0] = f32_to_f16(val);
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &dens_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
                offset: 0, bytes_per_row: Some(g * 8), rows_per_image: Some(g),
            },
            wgpu::Extent3d { width: g, height: g, depth_or_array_layers: 1 },
        );
    }

    // ---- Uniform buffer ----
    let mut sim_params = SimParams {
        grid_size: GRID_SIZE, mouse_down: 0, dt: 0.016, viscosity: 0.0001,
        dissipation: 0.998, add_strength: 2.0, mouse_pos: [128.0, 128.0],
        mouse_delta: [0.0, 0.0], radius: 35.0, _pad0: 0.0, _pad1: [0.0; 4],
    };

    let param_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&sim_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // ---- Bind group layouts ----
    let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute_bgl"),
        entries: &(0..8u32).map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if i == 0 {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                }
            } else {
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }
            },
            count: None,
        }).collect::<Vec<_>>(),
    });

    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
                count: None,
            },
        ],
    });

    // ---- Bind groups ----
    let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bg"), layout: &compute_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&vel_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&vel_tmp_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&dens_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&dens_tmp_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&press_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&press_tmp_view) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&div_view) },
        ],
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });

    let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"), layout: &render_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dens_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&vel_view) },
        ],
    });

    // ---- Pipelines ----
    let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&compute_bgl], push_constant_ranges: &[],
    });
    let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&render_bgl], push_constant_ranges: &[],
    });

    let make_compute = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry), layout: Some(&compute_pl),
            module: &compute_shader, entry_point: entry,
        })
    };

    let add_source_pipe = make_compute("add_source");
    let advect_vel_pipe = make_compute("advect_vel");
    let copy_vel_pipe = make_compute("copy_vel");
    let advect_dens_pipe = make_compute("advect_dens");
    let copy_dens_pipe = make_compute("copy_dens");
    let divergence_pipe = make_compute("compute_divergence");
    let pressure_a_pipe = make_compute("pressure_jacobi_a");
    let pressure_b_pipe = make_compute("pressure_jacobi_b");
    let gradient_pipe = make_compute("subtract_gradient");

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"), layout: Some(&render_pl),
        vertex: wgpu::VertexState {
            module: &render_shader, entry_point: "vs_fullscreen", buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader, entry_point: "fs_draw",
            targets: &[Some(wgpu::ColorTargetState {
                format, blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // ---- State ----
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut window_size = window.inner_size();
    let mut frame_count: u64 = 0;

    eprintln!("Starting event loop...");

    // ---- Event loop ----
    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event: ref win_event, .. } => match win_event {
                WindowEvent::CloseRequested => target.exit(),

                WindowEvent::Resized(new_size) => {
                    if new_size.width > 0 && new_size.height > 0 {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        window_size = *new_size;
                        surface.configure(&device, &config);
                    }
                }

                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    sim_params.mouse_down = if *state == ElementState::Pressed { 1 } else { 0 };
                    if *state == ElementState::Released {
                        last_mouse = None;
                        sim_params.mouse_delta = [0.0, 0.0];
                    }
                    eprintln!("CLICK: down={}", sim_params.mouse_down);
                }

                WindowEvent::CursorMoved { position, .. } => {
                    let sx = GRID_SIZE as f32 / window_size.width.max(1) as f32;
                    let sy = GRID_SIZE as f32 / window_size.height.max(1) as f32;
                    let mx = position.x as f32 * sx;
                    let my = position.y as f32 * sy;

                    if let Some((px, py)) = last_mouse {
                        sim_params.mouse_delta = [mx - px, my - py];
                    }
                    sim_params.mouse_pos = [mx, my];
                    last_mouse = Some((mx, my));
                }

                WindowEvent::Touch(touch) => {
                    let sx = GRID_SIZE as f32 / window_size.width.max(1) as f32;
                    let sy = GRID_SIZE as f32 / window_size.height.max(1) as f32;
                    let mx = touch.location.x as f32 * sx;
                    let my = touch.location.y as f32 * sy;

                    match touch.phase {
                        TouchPhase::Started => {
                            sim_params.mouse_down = 1;
                            sim_params.mouse_pos = [mx, my];
                            last_mouse = Some((mx, my));
                            eprintln!("TOUCH START ({:.0}, {:.0})", mx, my);
                        }
                        TouchPhase::Moved => {
                            if let Some((px, py)) = last_mouse {
                                sim_params.mouse_delta = [mx - px, my - py];
                            }
                            sim_params.mouse_pos = [mx, my];
                            last_mouse = Some((mx, my));
                        }
                        TouchPhase::Ended | TouchPhase::Cancelled => {
                            sim_params.mouse_down = 0;
                            last_mouse = None;
                            sim_params.mouse_delta = [0.0, 0.0];
                            eprintln!("TOUCH END");
                        }
                    }
                }

                WindowEvent::RedrawRequested => {
                    frame_count += 1;
                    if frame_count % 120 == 0 {
                        eprintln!(
                            "[frame {}] down={} pos=[{:.0},{:.0}] delta=[{:.1},{:.1}]",
                            frame_count, sim_params.mouse_down,
                            sim_params.mouse_pos[0], sim_params.mouse_pos[1],
                            sim_params.mouse_delta[0], sim_params.mouse_delta[1],
                        );
                    }

                    queue.write_buffer(&param_buffer, 0, bytemuck::bytes_of(&sim_params));

                    let frame = match surface.get_current_texture() {
                        Ok(f) => f,
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            surface.configure(&device, &config);
                            return;
                        }
                        Err(e) => {
                            eprintln!("Surface error: {:?}", e);
                            return;
                        }
                    };

                    let view = frame.texture.create_view(&Default::default());
                    let mut encoder = device.create_command_encoder(&Default::default());

                    // Compute pass
                    {
                        let mut c = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("sim"), timestamp_writes: None,
                        });
                        c.set_bind_group(0, &compute_bg, &[]);

                        c.set_pipeline(&add_source_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                        c.set_pipeline(&advect_vel_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                        c.set_pipeline(&copy_vel_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                        c.set_pipeline(&advect_dens_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                        c.set_pipeline(&copy_dens_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                        c.set_pipeline(&divergence_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                        for _ in 0..20 {
                            c.set_pipeline(&pressure_a_pipe);
                            c.dispatch_workgroups(wg.0, wg.1, 1);
                            c.set_pipeline(&pressure_b_pipe);
                            c.dispatch_workgroups(wg.0, wg.1, 1);
                        }
                        c.set_pipeline(&gradient_pipe);
                        c.dispatch_workgroups(wg.0, wg.1, 1);
                    }

                    // Render pass
                    {
                        let mut r = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("render"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view, resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        r.set_pipeline(&render_pipeline);
                        r.set_bind_group(0, &render_bg, &[]);
                        r.draw(0..3, 0..1);
                    }

                    queue.submit(Some(encoder.finish()));
                    frame.present();
                    sim_params.mouse_delta = [0.0, 0.0];
                }

                _ => {}
            },

            Event::AboutToWait => {
                window.request_redraw();
            }

            _ => {}
        }
    }).ok();
}