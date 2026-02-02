# wgpu-fluid

Real-time 2D Eulerian fluid simulation running on the GPU. Built with Rust, wgpu, and hand-written WGSL compute shaders.

## What it does

Simulates incompressible fluid dynamics on a 256×256 grid entirely on the GPU. Click and drag to inject dye and velocity — the simulation handles advection, pressure projection, and dissipation in real time.

The rendering maps fluid density and velocity to a rainbow color palette with a glow effect. Flow direction determines hue, speed determines saturation, and dense regions bloom to white.

## How it works

The simulation runs as a sequence of compute shader passes each frame:

1. **Source injection** — mouse input adds velocity and dye density with a Gaussian falloff
2. **Advection** — density and velocity fields are transported along the velocity field via bilinear interpolation
3. **Pressure solve** — divergence is computed, then 20 Jacobi iterations solve the pressure Poisson equation
4. **Projection** — the pressure gradient is subtracted from velocity to enforce incompressibility

All fields (velocity, density, pressure, divergence) are stored as `Rgba16Float` storage textures with read-write access. A fullscreen triangle pass samples the density and velocity textures to produce the final image.

## Project structure

```
├── Cargo.toml
├── fluid.wgsl        # All compute kernels (advection, pressure, projection)
└── src/
    └── main.rs        # GPU setup, render shader (inline), event loop
```

The render shader is defined inline in `main.rs` as a separate shader module from the compute shader. This avoids bind group layout conflicts between the compute and render pipelines.

## Dependencies

| Crate       | Purpose                        |
|-------------|--------------------------------|
| wgpu 0.19   | WebGPU API for Rust            |
| winit 0.29  | Window creation and input      |
| pollster    | Minimal async executor         |
| bytemuck    | Safe byte casting for uniforms |
| env_logger  | Debug logging                  |

## Requirements

- Rust toolchain (stable)
- Vulkan-capable GPU driver (or llvmpipe for software rendering)
- On WSL2: runs via XWayland (the app forces X11 backend automatically)

## Build and run

```
cargo run
```

Release build for better performance:

```
cargo run --release
```

## Controls

- **Left click + drag** — inject dye and velocity
- **Close window** — exit

## Configuration

Simulation parameters are hardcoded in `main.rs` as `SimParams`:

| Parameter      | Default | Effect                              |
|----------------|---------|-------------------------------------|
| `grid_size`    | 256     | Simulation resolution               |
| `dt`           | 0.016   | Timestep                            |
| `dissipation`  | 0.998   | How quickly density/velocity fade   |
| `add_strength` | 2.0     | Dye injection intensity             |
| `radius`       | 35.0    | Brush radius in grid cells          |
| `viscosity`    | 0.0001  | Fluid viscosity (currently unused)  |

## Known limitations

- No vorticity confinement — swirls dissipate faster than they would in a real fluid
- Requires `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` (native-only wgpu feature for read-write storage textures)
- Software rendering (llvmpipe) works but is slower than hardware Vulkan