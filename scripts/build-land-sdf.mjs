// Downloads the Natural Earth 1:10m land and lakes GeoJSON and bakes a signed
// distance field of the land (lakes count as water) into a cubemap for the
// app. Pipeline: scanline-rasterize the polygons into a binary equirect mask
// (their native coordinate space, so no clipping or projection is needed),
// resample each gnomonic cube face from it with a gutter so the distance
// transform sees across face seams, run a signed Euclidean distance transform
// per face, and encode the narrow band to bytes.
//
// Layout (little-endian):
//   u32    face size in texels
//   u8     6 faces in WebGPU layer order (+X,-X,+Y,-Y,+Z,-Z), row-major;
//          128 ~= coastline, full range spans +-BAND face texels of distance
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const NATURAL_EARTH_BASE =
  "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson";
const MAX_LAKE_SCALERANK = 4; // 0 = Great Lakes tier, 9 = everything in the 10m dataset
const EQUIRECT_WIDTH = 16384; // transient rasterization buffer, never shipped
const EQUIRECT_HEIGHT = EQUIRECT_WIDTH / 2;
const FACE_SIZE = 1024;
const GUTTER = 64; // face texels sampled past each edge for seam-correct distances
const SUPERSAMPLE = 2; // faces are resampled and distance-transformed at this multiple
const BAND = 4; // half-range of the encoded distance, in face texels
const outPath = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data", "land-sdf.bin");

async function downloadGeojson(layer) {
  const url = `${NATURAL_EARTH_BASE}/${layer}.geojson`;
  console.log(`Downloading ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function geometryToRings(geometry) {
  switch (geometry.type) {
    case "Polygon":
      return geometry.coordinates;
    case "MultiPolygon":
      return geometry.coordinates.flat();
    default:
      return [];
  }
}

// Even-odd scanline fill of the features' rings into a binary equirect mask
// (row 0 = +90째 latitude, column 0 = -180째 longitude, half-texel centers).
// Holes and overlaps cancel via the global even-odd rule.
function rasterizeMask(features) {
  const crossingsPerRow = Array.from({ length: EQUIRECT_HEIGHT }, () => []);
  for (const feature of features) {
    for (const ring of geometryToRings(feature.geometry)) {
      for (let i = 0; i + 1 < ring.length; i++) {
        // Continuous row coordinates; row r's center sits at r + 0.5.
        const y0 = ((90 - ring[i][1]) / 180) * EQUIRECT_HEIGHT;
        const y1 = ((90 - ring[i + 1][1]) / 180) * EQUIRECT_HEIGHT;
        if (y0 === y1) continue;
        // Half-open span [min, max) so shared vertices aren't double-counted.
        const first = Math.max(0, Math.ceil(Math.min(y0, y1) - 0.5));
        const last = Math.min(EQUIRECT_HEIGHT - 1, Math.ceil(Math.max(y0, y1) - 0.5) - 1);
        for (let r = first; r <= last; r++) {
          const t = (r + 0.5 - y0) / (y1 - y0);
          const longitude = ring[i][0] + t * (ring[i + 1][0] - ring[i][0]);
          crossingsPerRow[r].push(((longitude + 180) / 360) * EQUIRECT_WIDTH);
        }
      }
    }
  }
  const mask = new Uint8Array(EQUIRECT_WIDTH * EQUIRECT_HEIGHT);
  for (let r = 0; r < EQUIRECT_HEIGHT; r++) {
    const xs = crossingsPerRow[r].sort((a, b) => a - b);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const from = Math.max(0, Math.ceil(xs[i] - 0.5));
      const to = Math.min(EQUIRECT_WIDTH - 1, Math.ceil(xs[i + 1] - 0.5) - 1);
      mask.fill(1, r * EQUIRECT_WIDTH + from, r * EQUIRECT_WIDTH + to + 1);
    }
  }
  return mask;
}

// Direction through a cube face point, WebGPU layer order and orientation
// (s right, t down on the face image, both in [-1, 1] at the face edges).
function faceDirection(face, s, t) {
  switch (face) {
    case 0: return [1, -t, -s]; // +X
    case 1: return [-1, -t, s]; // -X
    case 2: return [s, 1, t]; // +Y
    case 3: return [s, -1, -t]; // -Y
    case 4: return [s, -t, 1]; // +Z
    case 5: return [-s, -t, -1]; // -Z
  }
}

function sampleEquirect(mask, direction) {
  const [x, y, z] = direction;
  const length = Math.sqrt(x * x + y * y + z * z);
  // Matches latLonToXyz: +Y north, longitude = atan2(-z, x).
  const latitude = Math.asin(y / length);
  const longitude = Math.atan2(-z, x);
  let column = Math.floor((longitude / (2 * Math.PI) + 0.5) * EQUIRECT_WIDTH);
  column = ((column % EQUIRECT_WIDTH) + EQUIRECT_WIDTH) % EQUIRECT_WIDTH;
  const row = Math.min(
    EQUIRECT_HEIGHT - 1,
    Math.max(0, Math.floor((0.5 - latitude / Math.PI) * EQUIRECT_HEIGHT)),
  );
  return mask[row * EQUIRECT_WIDTH + column];
}

// Felzenszwalb & Huttenlocher 1D squared distance transform (lower envelope
// of parabolas); f is the input row, d the output, v/z scratch.
function distanceTransform1d(f, n, d, v, z) {
  const INF = 1e20;
  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = INF;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

// Squared distance from every grid cell to the nearest cell where on() is
// truthy; separable 1D transforms over columns then rows.
function squaredDistances(on, size) {
  const INF = 1e20;
  const grid = new Float64Array(size * size);
  const f = new Float64Array(size);
  const d = new Float64Array(size);
  const v = new Int32Array(size);
  const z = new Float64Array(size + 1);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) f[y] = on[y * size + x] ? 0 : INF;
    distanceTransform1d(f, size, d, v, z);
    for (let y = 0; y < size; y++) grid[y * size + x] = d[y];
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) f[x] = grid[y * size + x];
    distanceTransform1d(f, size, d, v, z);
    for (let x = 0; x < size; x++) grid[y * size + x] = d[x];
  }
  return grid;
}

const land = await downloadGeojson("ne_10m_land");
const lakes = await downloadGeojson("ne_10m_lakes");

console.log("Rasterizing land");
const mask = rasterizeMask(land.features);
console.log("Rasterizing lakes");
const lakeMask = rasterizeMask(
  lakes.features.filter((feature) => feature.properties.scalerank <= MAX_LAKE_SCALERANK),
);
let landTexels = 0;
for (let i = 0; i < mask.length; i++) {
  mask[i] = mask[i] & ~lakeMask[i];
  landTexels += mask[i];
}
console.log(`Land fraction: ${((landTexels / mask.length) * 100).toFixed(1)}%`);

const output = new Uint8Array(4 + 6 * FACE_SIZE * FACE_SIZE);
new DataView(output.buffer).setUint32(0, FACE_SIZE, true);

const padded = (FACE_SIZE + 2 * GUTTER) * SUPERSAMPLE;
// Hi-res grid index -> face-plane coordinate in [-1, 1] at the face edges.
const planeCoord = (i) => ((((i + 0.5) / SUPERSAMPLE - GUTTER) / FACE_SIZE) * 2 - 1);

for (let face = 0; face < 6; face++) {
  console.log(`Face ${face}: resampling and distance transform`);
  const faceMask = new Uint8Array(padded * padded);
  for (let i = 0; i < padded; i++) {
    const t = planeCoord(i);
    for (let j = 0; j < padded; j++) {
      faceMask[i * padded + j] = sampleEquirect(mask, faceDirection(face, planeCoord(j), t));
    }
  }
  const toLand = squaredDistances(faceMask, padded);
  const toWater = squaredDistances(
    faceMask.map((m) => 1 - m),
    padded,
  );
  // Box-downsample the signed distance to the final face, crop the gutter,
  // convert hi-res pixels to face texels, and encode the +-BAND narrow band.
  const offset = 4 + face * FACE_SIZE * FACE_SIZE;
  for (let fy = 0; fy < FACE_SIZE; fy++) {
    for (let fx = 0; fx < FACE_SIZE; fx++) {
      let sum = 0;
      for (let sy = 0; sy < SUPERSAMPLE; sy++) {
        for (let sx = 0; sx < SUPERSAMPLE; sx++) {
          const idx =
            ((fy + GUTTER) * SUPERSAMPLE + sy) * padded + (fx + GUTTER) * SUPERSAMPLE + sx;
          sum += faceMask[idx] ? Math.sqrt(toWater[idx]) : -Math.sqrt(toLand[idx]);
        }
      }
      const texels = sum / (SUPERSAMPLE * SUPERSAMPLE) / SUPERSAMPLE;
      output[offset + fy * FACE_SIZE + fx] = Math.min(
        255,
        Math.max(0, Math.round(127.5 + (texels / BAND) * 127.5)),
      );
    }
  }
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, output);
console.log(
  `Wrote ${outPath}: 6 faces of ${FACE_SIZE}x${FACE_SIZE}, ` +
    `${(output.byteLength / 1024 / 1024).toFixed(2)} MB`,
);
