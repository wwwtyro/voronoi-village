// Spherical Voronoi cells of the K most populous cities, one cell per
// invocation (after Ray et al. 2018, "Meshless Voronoi on the GPU"): each
// thread starts from a spherical cap around its site and clips it by the
// bisector plane of every nearby site, walking a spatial grid outward until
// the security radius guarantees no farther site can cut the cell. Surviving
// edges are written as segment endpoint pairs via an atomic counter; a second
// entry point converts the counter into the indirect args of the instanced
// quad draw that renders them.
//
// With a nonzero population weight the diagram is a spherical power
// (Laguerre) diagram: bisector planes are offset from the origin and edges
// lie on small circles. relaxMain additionally Lloyd-relaxes virtual site
// positions toward their cells' centroids; city dots and tooltips elsewhere
// keep the true positions.
//
// Accepted artifacts of the budgets below: cells needing more than
// MAX_VERTS vertices skip the overflowing clip and render slightly too
// large; at extreme thresholds (fewer than ~20 visible cities) true cells
// can exceed the initial cap and get truncated to it; f32 clipping is not
// exact, so near-degenerate site configurations may show hairline edge
// mismatches between adjacent cells. At nonzero weight a sufficiently
// outweighed city's cell vanishes entirely while its dot remains.

const ROWS = 128u; // latitude rows in the spatial grid; keep in sync with voronoi.ts
const MAX_VERTS = 32u;
const CAP_RADIUS = 1.3962634; // 80 degrees
// ~1 degree pieces keep the chords within 4e-5 of the sphere: sub-pixel
// against the analytic surface at any zoom, and well inside the sphere
// shader's DEPTH_OFFSET.
const SUBDIV_RADIANS = 0.017453293;
const MAX_SEGMENTS = 32u;
// Maximum relaxed-site displacement from the true position, in radians; keeps
// the spatial grid (built from true positions) valid given the search slack.
// Keep in sync with voronoi.ts.
const S_MAX = 0.05;
const INVALID = 0xffffffffu;
const PI = 3.14159265;
const TWO_PI = 6.28318531;
const HALF_PI = 1.57079633;
const DLAT = PI / f32(ROWS);

struct Uniforms {
  k: u32,
  capacity: u32, // outVertices capacity, in vertices
  hovered: u32, // city index for highlightMain; >= k means none
  alpha: f32, // population weight scale; 0 is the unweighted diagram
  slack: f32, // search-limit allowance for stale grid bins; 2 * S_MAX when relaxing
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read> rowStart: array<u32>;
@group(0) @binding(3) var<storage, read> binOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> binEntries: array<u32>;
@group(0) @binding(5) var<storage, read_write> outVertices: array<f32>;
@group(0) @binding(6) var<storage, read_write> vertexCounter: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> indirectArgs: array<u32, 4>;
// The hovered cell as bounding half-spaces for the sphere shader's highlight:
// inside iff dot(p, plane.xyz) + plane.w >= 0 for every plane. At most
// MAX_VERTS bisector edges plus the initial-cap constraint. Keep the struct in
// sync with sphere.wgsl.
struct HighlightCell {
  count: u32,
  planes: array<vec4f, 33>,
}
@group(0) @binding(8) var<storage, read_write> highlightCell: HighlightCell;
// log10(population / 500) per city, parallel to positions (so sorted
// descending); the power-diagram weight of city i is alpha * logPop[i].
@group(0) @binding(9) var<storage, read> logPop: array<f32>;
// Lloyd relaxation I/O (relaxMain only): cells are built from the current
// positions binding and the moved sites land in outPositions; the S_MAX clamp
// is measured against the true city positions.
@group(0) @binding(10) var<storage, read_write> outPositions: array<vec4f>;
@group(0) @binding(11) var<storage, read> originalPositions: array<vec4f>;

var<private> selfIndex: u32;
var<private> site: vec3f;
var<private> siteWeight: f32;
// alpha times the visible prefix's logPop range: the most any rival's weight
// advantage over this site can be.
var<private> weightSpread: f32;
var<private> siteLat: f32;
var<private> siteLon: f32;
// The cell polygon: unit vectors in cyclic order. cellIds[e] is the index of
// the city whose bisector contains the edge cellVerts[e] -> cellVerts[e+1],
// or INVALID for edges left over from the initial cap.
var<private> cellVerts: array<vec3f, MAX_VERTS>;
var<private> cellIds: array<u32, MAX_VERTS>;
var<private> cellCount: u32;
var<private> cosSearchLimit: f32;
// Per-row window of already-visited columns, in unwrapped column indices;
// lo > hi means the row is untouched.
var<private> visitedLo: array<i32, ROWS>;
var<private> visitedHi: array<i32, ROWS>;

fn initCap() {
  let pole = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(site.y) > 0.9);
  let u = normalize(cross(site, pole));
  let v = cross(site, u);
  let c = cos(CAP_RADIUS);
  let s = sin(CAP_RADIUS);
  cellVerts[0] = c * site + s * u;
  cellVerts[1] = c * site + s * v;
  cellVerts[2] = c * site - s * u;
  cellVerts[3] = c * site - s * v;
  for (var e = 0u; e < 4u; e++) {
    cellIds[e] = INVALID;
  }
  cellCount = 4u;
}

// The weighted (power-diagram) bisector between site and city j: the cell
// keeps the side with dot(p, plane.xyz) + plane.w >= 0. At zero alpha this is
// the perpendicular-bisector plane through the origin.
fn bisector(j: u32) -> vec4f {
  let toSite = site - positions[j].xyz;
  let l = length(toSite);
  return vec4f(toSite / l, (siteWeight - uniforms.alpha * logPop[j]) / l);
}

// The plane whose circle contains polygon edge e (a -> b): the neighbor's
// bisector when the edge has one, else the great-circle plane through the
// endpoints (initial-cap edges are great arcs by construction).
fn edgePlane(e: u32, a: vec3f, b: vec3f) -> vec4f {
  let id = cellIds[e];
  if (id != INVALID) {
    return bisector(id);
  }
  return vec4f(normalize(cross(a, b)), 0.0);
}

// Where polygon edge e (a -> b) crosses the clipping plane: the exact
// intersection of the edge's plane, the clipping plane, and the unit sphere.
// The normalized chord point is that intersection for planes through the
// origin but drifts off offset (weighted) planes, so it serves as the
// root-picking hint and the degenerate fallback only.
fn crossingVertex(e: u32, a: vec3f, b: vec3f, plane: vec4f, da: f32, db: f32) -> vec3f {
  let hint = normalize(mix(a, b, da / (da - db)));
  let ep = edgePlane(e, a, b);
  let c = dot(ep.xyz, plane.xyz);
  let denom = 1.0 - c * c;
  if (denom < 1e-9) {
    return hint; // near-parallel planes
  }
  let p0 = ((plane.w * c - ep.w) * ep.xyz + (ep.w * c - plane.w) * plane.xyz) / denom;
  let h2 = 1.0 - dot(p0, p0);
  if (h2 <= 0.0) {
    return hint; // the circles barely miss; numerical degeneracy
  }
  // The two roots sit at p0 +- sqrt(h2) * d with p0 orthogonal to d, so the
  // hint's side of d picks the root between a and b.
  let d = cross(ep.xyz, plane.xyz) * inverseSqrt(denom);
  return p0 + select(-1.0, 1.0, dot(hint, d) >= 0.0) * sqrt(h2) * d;
}

// Clip the cell by the weighted bisector of site and city j.
fn clipCell(j: u32) {
  let toSite = site - positions[j].xyz;
  if (dot(toSite, toSite) < 1e-12) {
    return; // co-located cities; both keep overlapping cells
  }
  let plane = bisector(j);
  var newVerts: array<vec3f, MAX_VERTS>;
  var newIds: array<u32, MAX_VERTS>;
  var m = 0u;
  for (var e = 0u; e < cellCount; e++) {
    let a = cellVerts[e];
    let b = cellVerts[(e + 1u) % cellCount];
    let da = dot(a, plane.xyz) + plane.w;
    let db = dot(b, plane.xyz) + plane.w;
    if (da >= 0.0) {
      if (m < MAX_VERTS) {
        newVerts[m] = a;
        newIds[m] = cellIds[e];
      }
      m += 1u;
      if (db < 0.0) {
        if (m < MAX_VERTS) {
          newVerts[m] = crossingVertex(e, a, b, plane, da, db);
          newIds[m] = j;
        }
        m += 1u;
      }
    } else if (db >= 0.0) {
      if (m < MAX_VERTS) {
        newVerts[m] = crossingVertex(e, a, b, plane, da, db);
        newIds[m] = cellIds[e];
      }
      m += 1u;
    }
  }
  if (m > MAX_VERTS) {
    return; // over budget: skip this clip
  }
  if (m == 0u || (m < 3u && dot(site, plane.xyz) + plane.w < 0.0)) {
    cellCount = 0u; // the cell is swallowed at this weight
    return;
  }
  if (m < 3u) {
    return; // numerically degenerate sliver: skip this clip
  }
  for (var e = 0u; e < m; e++) {
    cellVerts[e] = newVerts[e];
    cellIds[e] = newIds[e];
  }
  cellCount = m;
}

fn scanBin(bin: u32) {
  let end = binOffsets[bin + 1u];
  for (var t = binOffsets[bin]; t < end; t++) {
    let j = binEntries[t];
    if (j >= uniforms.k) {
      break; // entries are ascending; the rest are below the threshold
    }
    if (j == selfIndex) {
      continue;
    }
    if (dot(site, positions[j].xyz) < cosSearchLimit) {
      continue; // beyond the security limit (see searchLimit): cannot cut
    }
    clipCell(j);
  }
}

fn wrapCol(col: i32, colCount: i32) -> u32 {
  return u32(((col % colCount) + colCount) % colCount);
}

// Visit every bin intersecting the spherical disc of the given radius around
// the site that has not been visited at a smaller radius. Row and column
// windows are conservative (one extra row/column) so the disc is always fully
// covered; visiting extra bins is harmless, missing one would be a bug.
fn expandWindows(radius: f32) {
  let rowLo = max(0, i32(floor((siteLat - radius + HALF_PI) / DLAT)) - 1);
  let rowHi = min(i32(ROWS) - 1, i32(floor((siteLat + radius + HALF_PI) / DLAT)) + 1);
  for (var row = rowLo; row <= rowHi; row++) {
    let firstBin = rowStart[u32(row)];
    let colCount = i32(rowStart[u32(row) + 1u] - firstBin);
    let prevLo = visitedLo[row];
    let prevHi = visitedHi[row];
    if (prevHi - prevLo + 1 >= colCount) {
      continue; // row fully visited
    }
    // Longitude half-window covering the disc within this row's latitude
    // band: sin(dLon) = sin(radius) / cos(lat), full row once the disc wraps
    // or reaches a pole.
    let lat0 = f32(row) * DLAT - HALF_PI;
    let cosBand = min(cos(lat0), cos(lat0 + DLAT));
    var full = radius >= HALF_PI || sin(radius) >= cosBand;
    var halfCols = 0;
    if (!full) {
      let halfLon = asin(sin(radius) / cosBand);
      halfCols = i32(ceil(halfLon * f32(colCount) / TWO_PI)) + 1;
      full = 2 * halfCols + 1 >= colCount;
    }
    let hasPrev = prevHi >= prevLo;
    if (full) {
      if (hasPrev) {
        // Complement of the previously visited window, wrapped.
        for (var c = prevHi + 1; c <= prevLo + colCount - 1; c++) {
          scanBin(firstBin + wrapCol(c, colCount));
        }
      } else {
        for (var c = 0; c < colCount; c++) {
          scanBin(firstBin + u32(c));
        }
      }
      visitedLo[row] = 0;
      visitedHi[row] = colCount - 1;
    } else {
      let center = clamp(i32(floor((siteLon + PI) / TWO_PI * f32(colCount))), 0, colCount - 1);
      let newLo = center - halfCols;
      let newHi = center + halfCols;
      if (hasPrev) {
        for (var c = newLo; c < prevLo; c++) {
          scanBin(firstBin + wrapCol(c, colCount));
        }
        for (var c = prevHi + 1; c <= newHi; c++) {
          scanBin(firstBin + wrapCol(c, colCount));
        }
        visitedLo[row] = min(newLo, prevLo);
        visitedHi[row] = max(newHi, prevHi);
      } else {
        for (var c = newLo; c <= newHi; c++) {
          scanBin(firstBin + wrapCol(c, colCount));
        }
        visitedLo[row] = newLo;
        visitedHi[row] = newHi;
      }
    }
  }
}

fn writeSegment(base: u32, p0: vec3f, p1: vec3f) {
  let o = base * 3u;
  outVertices[o] = p0.x;
  outVertices[o + 1u] = p0.y;
  outVertices[o + 2u] = p0.z;
  outVertices[o + 3u] = p1.x;
  outVertices[o + 4u] = p1.y;
  outVertices[o + 5u] = p1.z;
}

fn emitEdges() {
  for (var e = 0u; e < cellCount; e++) {
    let id = cellIds[e];
    if (id != INVALID && id <= selfIndex) {
      continue; // the shared edge is emitted by the lower-index cell only
    }
    let a = cellVerts[e];
    let b = cellVerts[(e + 1u) % cellCount];
    let arc = acos(clamp(dot(a, b), -1.0, 1.0));
    if (arc <= SUBDIV_RADIANS) {
      let base = atomicAdd(&vertexCounter, 2u);
      if (base + 2u <= uniforms.capacity) {
        writeSegment(base, a, b);
      }
      continue;
    }
    // Subdivision points lie on the edge's circle: weighted bisector edges
    // follow their plane's small circle (center -w * n, radius sqrt(1 - w^2));
    // for unweighted and cap edges w = 0 and the projection reduces to
    // normalize(mix(...)) on the great circle through a and b. The segment
    // count is scaled by 1/r because a chord subtends a wider angle on a
    // smaller circle.
    let plane = edgePlane(e, a, b);
    let center = -plane.w * plane.xyz;
    let r = sqrt(max(1.0 - plane.w * plane.w, 1e-6));
    let segments = clamp(u32(ceil(arc / (SUBDIV_RADIANS * max(r, 0.05)))), 1u, MAX_SEGMENTS);
    let base = atomicAdd(&vertexCounter, segments * 2u);
    if (base + segments * 2u > uniforms.capacity) {
      continue;
    }
    var prev = a;
    for (var s = 1u; s <= segments; s++) {
      var p = b;
      if (s < segments) {
        let chord = mix(a, b, f32(s) / f32(segments));
        let onPlane = chord - (dot(chord, plane.xyz) + plane.w) * plane.xyz;
        p = center + r * normalize(onPlane - center);
      }
      writeSegment(base + (s - 1u) * 2u, prev, p);
      prev = p;
    }
  }
}

// No site beyond this angular distance can cut the cell: a cell point p that
// some rival j claims satisfies dot(p, sj) > cos(maxAngle) - weightSpread,
// putting j within acos of that from p, which is itself within maxAngle of
// the site. slack covers grid bins gone stale under relaxation (every site
// stays within S_MAX of the position it was binned at). At zero weight and
// slack this is the classic 2 * maxAngle security radius.
fn searchLimit(maxAngle: f32) -> f32 {
  return maxAngle + acos(clamp(cos(maxAngle) - weightSpread, -1.0, 1.0)) + uniforms.slack;
}

// Construct the Voronoi cell of the given city into cellVerts/cellIds.
fn buildCell(index: u32) {
  selfIndex = index;
  site = positions[index].xyz;
  siteWeight = uniforms.alpha * logPop[index];
  weightSpread = uniforms.alpha * (logPop[0] - logPop[uniforms.k - 1u]);
  siteLat = asin(clamp(site.y, -1.0, 1.0));
  siteLon = atan2(-site.z, site.x);
  initCap();
  for (var r = 0u; r < ROWS; r++) {
    visitedLo[r] = 1;
    visitedHi[r] = 0;
  }
  var maxAngle = CAP_RADIUS;
  var limit = searchLimit(maxAngle);
  cosSearchLimit = select(cos(limit), -2.0, limit >= PI);
  // Grow the search disc one grid row at a time. A site beyond the security
  // limit cannot clip the cell, and the cell only shrinks, so once the
  // visited radius passes that limit the cell is final; the radius >= pi
  // case always terminates the loop.
  for (var s = 1u; s <= ROWS + 2u; s++) {
    let radius = f32(s) * DLAT;
    expandWindows(radius);
    maxAngle = 0.0;
    for (var e = 0u; e < cellCount; e++) {
      maxAngle = max(maxAngle, acos(clamp(dot(site, cellVerts[e]), -1.0, 1.0)));
    }
    limit = searchLimit(maxAngle);
    cosSearchLimit = select(cos(limit), -2.0, limit >= PI);
    if (radius >= limit || radius >= PI) {
      break;
    }
  }
}

@compute @workgroup_size(64)
fn voronoiMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= uniforms.k) {
    return;
  }
  buildCell(gid.x);
  emitEdges();
}

// Recomputes the hovered cell alone and publishes its bounding planes for the
// sphere shader's per-fragment inside test.
@compute @workgroup_size(1)
fn highlightMain() {
  if (uniforms.hovered >= uniforms.k) {
    highlightCell.count = 0u;
    return;
  }
  buildCell(uniforms.hovered);
  if (cellCount == 0u) {
    highlightCell.count = 0u; // swallowed at this weight: nothing to tint
    return;
  }
  var count = 0u;
  for (var e = 0u; e < cellCount; e++) {
    if (cellIds[e] != INVALID) {
      highlightCell.planes[count] = bisector(cellIds[e]);
      count += 1u;
    }
  }
  // The initial cap bounds any edges never replaced by a bisector.
  highlightCell.planes[count] = vec4f(site, -cos(CAP_RADIUS));
  count += 1u;
  highlightCell.count = count;
}

// One Lloyd step: move each virtual site toward its cell's centroid, clamped
// to S_MAX of the city's true position so the spatial grid (built from true
// positions) stays valid under the search slack.
@compute @workgroup_size(64)
fn relaxMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= uniforms.k) {
    return;
  }
  buildCell(gid.x);
  var moved = site;
  if (cellCount >= 3u) {
    var sum = vec3f(0.0);
    for (var e = 0u; e < cellCount; e++) {
      let a = cellVerts[e];
      let b = cellVerts[(e + 1u) % cellCount];
      // Fan triangle (site, a, b): planar area times three times the
      // centroid; the constant factors cancel in the normalize.
      sum += length(cross(a - site, b - site)) * (site + a + b);
    }
    if (dot(sum, sum) > 1e-12) {
      moved = normalize(sum);
    }
  }
  let orig = originalPositions[gid.x].xyz;
  let c = clamp(dot(moved, orig), -1.0, 1.0);
  if (acos(c) > S_MAX) {
    // Pull back onto the S_MAX cap around the true position.
    let tangent = normalize(moved - c * orig);
    moved = cos(S_MAX) * orig + sin(S_MAX) * tangent;
  }
  outPositions[gid.x] = vec4f(moved, 0.0);
}

@compute @workgroup_size(1)
fn finalizeMain() {
  // 4 strip vertices per quad, one instance per segment (vertex pair).
  indirectArgs[0] = 4u;
  indirectArgs[1] = min(atomicLoad(&vertexCounter), uniforms.capacity) / 2u;
  indirectArgs[2] = 0u;
  indirectArgs[3] = 0u;
}
