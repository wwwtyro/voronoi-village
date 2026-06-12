// Analytic sphere: a cube proxy is rasterized and each fragment ray-traces
// the sphere, coloring the surface land or water from the signed-distance
// cubemap. The written depth comes from a slightly smaller sphere on the same
// ray, so geometry sitting exactly on the surface wins the depth test without
// any world-space offset that would shift its pixels relative to the terrain.

// World-space gap between the visible surface and the depth surface; surface
// geometry within this distance of the sphere renders on top.
const DEPTH_OFFSET = 0.002;

struct Camera {
  viewProjection: mat4x4f,
}

struct Sphere {
  // xyz: camera position, w: sphere radius.
  eyeRadius: vec4f,
  landColor: vec4f,
  waterColor: vec4f,
  highlightColor: vec4f,
}

// The hovered Voronoi cell as bounding half-spaces, written by the highlight
// compute pass; inside iff dot(p, plane.xyz) + plane.w >= 0 for every plane.
// Keep in sync with voronoi.wgsl.
struct HighlightCell {
  count: u32,
  planes: array<vec4f, 33>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> sphere: Sphere;
@group(0) @binding(2) var sdfTexture: texture_cube<f32>;
@group(0) @binding(3) var sdfSampler: sampler;
@group(0) @binding(4) var<storage, read> highlightCell: HighlightCell;

// Signed distance proxy to the hovered cell boundary: the minimum half-space
// distance, negative outside, -1 when nothing is hovered.
fn cellDistance(p: vec3f) -> f32 {
  var d = select(-1.0, 1e9, highlightCell.count > 0u);
  for (var i = 0u; i < highlightCell.count; i++) {
    let plane = highlightCell.planes[i];
    d = min(d, dot(p, plane.xyz) + plane.w);
  }
  return d;
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
}

@vertex
fn vertexMain(@location(0) position: vec3f) -> VertexOutput {
  var out: VertexOutput;
  out.position = camera.viewProjection * vec4f(position, 1.0);
  out.worldPosition = position;
  return out;
}

struct FragmentOutput {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fragmentMain(in: VertexOutput) -> FragmentOutput {
  let eye = sphere.eyeRadius.xyz;
  let radius = sphere.eyeRadius.w;
  let direction = normalize(in.worldPosition - eye);
  // Nearest ray/sphere intersection; same quadratic as pickSphere on the CPU.
  let od = dot(eye, direction);
  let discriminant = od * od - (dot(eye, eye) - radius * radius);
  // Clamping instead of branching keeps discarded (helper) invocations
  // producing finite positions, so neighbors' fwidth below stays valid.
  let halfChord = sqrt(max(discriminant, 0.0));
  let tNear = -od - halfChord;
  let tFar = -od + halfChord;
  let nearHit = eye + direction * tNear;
  let farHit = eye + direction * tFar;
  let dNear = textureSampleLevel(sdfTexture, sdfSampler, normalize(nearHit), 0.0).r - 0.5;
  let wNear = max(fwidth(dNear), 1e-6);
  let dFar = textureSampleLevel(sdfTexture, sdfSampler, normalize(farHit), 0.0).r - 0.5;
  let wFar = max(fwidth(dFar), 1e-6);
  let cNear = cellDistance(normalize(nearHit));
  let hNear = max(fwidth(cNear), 1e-6);
  let cFar = cellDistance(normalize(farHit));
  let hFar = max(fwidth(cFar), 1e-6);
  if (discriminant < 0.0 || tNear < 0.0) {
    discard;
  }
  // Depth surface: the inner sphere along the same ray. Past its limb the
  // clamp degrades to the closest-approach depth, where nothing is occluded.
  let innerRadius = radius - DEPTH_OFFSET;
  let innerDiscriminant = od * od - (dot(eye, eye) - innerRadius * innerRadius);
  let tInner = -od - sqrt(max(innerDiscriminant, 0.0));
  let clip = camera.viewProjection * vec4f(eye + direction * tInner, 1.0);
  // The hovered cell tints land only, so the highlight clips to the coastline
  // for free; it rides both hits, showing dimly on the far side like
  // everything else. The far surface shows through the near one at its alpha;
  // back-side geometry drawn earlier shows through both via the blend.
  let nearLand = mix(sphere.landColor, sphere.highlightColor, smoothstep(-hNear, hNear, cNear));
  let farLand = mix(sphere.landColor, sphere.highlightColor, smoothstep(-hFar, hFar, cFar));
  let nearSurface = mix(sphere.waterColor, nearLand, smoothstep(-wNear, wNear, dNear));
  let farSurface = mix(sphere.waterColor, farLand, smoothstep(-wFar, wFar, dFar));
  var out: FragmentOutput;
  out.color = vec4f(mix(farSurface.rgb, nearSurface.rgb, nearSurface.a), nearSurface.a);
  out.depth = clip.z / clip.w;
  return out;
}
