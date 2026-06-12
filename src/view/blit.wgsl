// Downsamples the supersampled scene texture to the canvas. At an exact 2:1
// ratio every destination pixel center lands on the corner between four
// source texels, so one bilinear sample is a perfect 2x2 box filter.

@group(0) @binding(0) var sourceSampler: sampler;
@group(0) @binding(1) var sourceTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // Fullscreen triangle.
  let corner = vec2f(
    f32((vertexIndex << 1u) & 2u) * 2.0 - 1.0,
    f32(vertexIndex & 2u) * 2.0 - 1.0,
  );
  var out: VertexOutput;
  out.position = vec4f(corner, 0.0, 1.0);
  out.uv = corner * vec2f(0.5, -0.5) + 0.5;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  return textureSample(sourceTexture, sourceSampler, in.uv);
}
