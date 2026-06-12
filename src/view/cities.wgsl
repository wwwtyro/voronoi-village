struct Uniforms {
  viewProjection: mat4x4f,
}

struct Params {
  viewport: vec2f,
  dotSizePixels: f32,
  unused: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> color: vec4f;
@group(0) @binding(2) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) position: vec4f,
  // Corner of the quad in [-1, 1]²; the fragment stage carves the circle.
  @location(0) local: vec2f,
}

@vertex
fn vertexMain(
  @location(0) center: vec3f,
  @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
  let corner = vec2f(
    select(-1.0, 1.0, (vertexIndex & 1u) == 1u),
    select(-1.0, 1.0, (vertexIndex & 2u) == 2u),
  );
  let clip = uniforms.viewProjection * vec4f(center, 1.0);
  var out: VertexOutput;
  // Offset in clip space by half the dot size in pixels, keeping the quad
  // screen-aligned at a constant pixel size regardless of depth.
  out.position = vec4f(clip.xy + corner * (params.dotSizePixels / params.viewport) * clip.w, clip.zw);
  out.local = corner;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  // The hard edge is smoothed by the supersampled downscale.
  if (dot(in.local, in.local) > 1.0) {
    discard;
  }
  return color;
}
