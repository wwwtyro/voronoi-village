// Line segments as instanced screen-space quads: WebGPU's line primitives
// are fixed at one fragment wide, so each segment instance is expanded to the
// requested pixel width perpendicular to its screen direction. Fragments over
// water (per the signed-distance cubemap) are faded out and discarded.

struct Uniforms {
  viewProjection: mat4x4f,
}

struct Params {
  viewport: vec2f,
  halfWidthPixels: f32,
  unused: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> color: vec4f;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var sdfTexture: texture_cube<f32>;
@group(0) @binding(4) var sdfSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
}

@vertex
fn vertexMain(
  @location(0) a: vec3f,
  @location(1) b: vec3f,
  @builtin(vertex_index) vertexIndex: u32,
) -> VertexOutput {
  let clipA = uniforms.viewProjection * vec4f(a, 1.0);
  let clipB = uniforms.viewProjection * vec4f(b, 1.0);
  let halfViewport = params.viewport * 0.5;
  let screenA = clipA.xy / clipA.w * halfViewport;
  let screenB = clipB.xy / clipB.w * halfViewport;
  let dir = screenB - screenA;
  // Degenerate on screen: zero normal collapses the quad to nothing.
  var normal = vec2f(0.0);
  if (dot(dir, dir) > 1e-12) {
    normal = normalize(vec2f(-dir.y, dir.x));
  }
  // Triangle-strip corners: A-, A+, B-, B+.
  let endpoint = select(clipA, clipB, (vertexIndex & 2u) == 2u);
  let side = select(-1.0, 1.0, (vertexIndex & 1u) == 1u);
  let offset = normal * side * params.halfWidthPixels / halfViewport;
  var out: VertexOutput;
  out.position = vec4f(endpoint.xy + offset * endpoint.w, endpoint.zw);
  // Constant across the quad's width, interpolates along the segment.
  out.world = select(a, b, (vertexIndex & 2u) == 2u);
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4f {
  let d = textureSampleLevel(sdfTexture, sdfSampler, normalize(in.world), 0.0).r - 0.5;
  // The fade spans ~1 screen pixel rather than the SDF band: these fragments
  // write depth, so a wide translucent band would mask the sphere behind it.
  let w = max(fwidth(d), 1e-6);
  let alpha = color.a * smoothstep(-w, w, d);
  if (alpha < 0.004) {
    discard;
  }
  return vec4f(color.rgb, alpha);
}
