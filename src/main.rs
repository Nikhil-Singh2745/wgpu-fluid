use std::mem;
use wgpu::util::DeviceExt;
use winit::{
    dpi::LogicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    grid_size: u32,
    mouse_down: u32,
    jacobi_iterations: u32,
    _pad0: u32,
    dt: f32,
    viscosity: f32,
    dissipation: f32,
    add_strength: f32,
    mouse_pos: [f32; 2],
    mouse_delta: [f32; 2],
    radius: f32,
    _pad1: [f32; 3],
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
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("WGPU Fluid")
        .with_inner_size(LogicalSize::new(800.0, 800.0))
        .build(&event_loop)
        .unwrap();
    //To do list : Implement fluid simulation logic 
}