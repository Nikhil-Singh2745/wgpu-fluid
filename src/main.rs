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
    _pad: [f32; 3],
}

fn create_storage_tex(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
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
    
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("WGPU Fluid Simulation")
        .with_inner_size(LogicalSize::new(800.0, 800.0))
        .build(&event_loop)
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    let surface = instance.create_surface(&window).unwrap();
    
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    }))
    .expect("Failed to find adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    let size = window.inner_size();
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fluid_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../fluid.wgsl").into()),
    });

    let grid_size: u32 = 256;
    let workgroups = ((grid_size + 7) / 8, (grid_size + 7) / 8);

    let (_vel, vel_view) = create_storage_tex(&device, grid_size);
    let (_vel_tmp, vel_tmp_view) = create_storage_tex(&device, grid_size);
    let (_dens, dens_view) = create_storage_tex(&device, grid_size);
    let (_dens_tmp, dens_tmp_view) = create_storage_tex(&device, grid_size);
    let (_press, press_view) = create_storage_tex(&device, grid_size);
    let (_press_tmp, press_tmp_view) = create_storage_tex(&device, grid_size);
    let (_div, div_view) = create_storage_tex(&device, grid_size);

    let params = SimParams {
        grid_size,
        mouse_down: 0,
        dt: 0.016,
        viscosity: 0.0001,
        dissipation: 0.995,
        add_strength: 1.0,
        mouse_pos: [128.0, 128.0],
        mouse_delta: [0.0, 0.0],
        radius: 20.0,
        _pad: [0.0; 3],
    };

    let param_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bg"),
        layout: &compute_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&vel_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&vel_tmp_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&dens_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&dens_tmp_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&press_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&press_tmp_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&div_view),
            },
        ],
    });

    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("render_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"),
        layout: &render_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&dens_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pl"),
        bind_group_layouts: &[&compute_bgl],
        push_constant_ranges: &[],
    });

    let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pl"),
        bind_group_layouts: &[&render_bgl],
        push_constant_ranges: &[],
    });

    let make_compute = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&compute_pl),
            module: &shader,
            entry_point: entry,
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
        label: Some("render_pipeline"),
        layout: Some(&render_pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_fullscreen",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_draw",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut sim_params = params;
    let mut last_mouse: Option<(f32, f32)> = None;
    let mut window_size = window.inner_size();

    event_loop
        .run(move |event, target| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            config.width = new_size.width;
                            config.height = new_size.height;
                            window_size = new_size;
                            surface.configure(&device, &config);
                        }
                    }
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    } => {
                        sim_params.mouse_down = if state == ElementState::Pressed { 1 } else { 0 };
                        if state == ElementState::Released {
                            last_mouse = None;
                            sim_params.mouse_delta = [0.0, 0.0];
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let scale_x = grid_size as f32 / window_size.width as f32;
                        let scale_y = grid_size as f32 / window_size.height as f32;
                        
                        let mx = position.x as f32 * scale_x;
                        let my = position.y as f32 * scale_y;
                        
                        if let Some((px, py)) = last_mouse {
                            sim_params.mouse_delta = [mx - px, my - py];
                        }
                        sim_params.mouse_pos = [mx, my];
                        last_mouse = Some((mx, my));
                    }
                    _ => {}
                },
                Event::AboutToWait => {
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

                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("fluid_sim"),
                            timestamp_writes: None,
                        });
                        cpass.set_bind_group(0, &compute_bg, &[]);

                        cpass.set_pipeline(&add_source_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);

                        cpass.set_pipeline(&advect_vel_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
                        cpass.set_pipeline(&copy_vel_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);

                        cpass.set_pipeline(&advect_dens_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
                        cpass.set_pipeline(&copy_dens_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);

                        cpass.set_pipeline(&divergence_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);

                        for _ in 0..20 {
                            cpass.set_pipeline(&pressure_a_pipe);
                            cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
                            cpass.set_pipeline(&pressure_b_pipe);
                            cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
                        }

                        cpass.set_pipeline(&gradient_pipe);
                        cpass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
                    }

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("render"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        rpass.set_pipeline(&render_pipeline);
                        rpass.set_bind_group(0, &render_bg, &[]);
                        rpass.draw(0..3, 0..1);
                    }

                    queue.submit(Some(encoder.finish()));
                    frame.present();
                    
                    // Reset delta after frame
                    sim_params.mouse_delta = [0.0, 0.0];
                }
                _ => {}
            }
        })
        .unwrap();
    }    