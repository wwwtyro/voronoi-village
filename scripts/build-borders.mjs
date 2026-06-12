// Downloads the Natural Earth 1:10m country and state/province boundary line
// GeoJSON (land boundaries only; coastlines are already drawn by the land
// SDF) and flattens them into polylines for the app. Long segments are
// subdivided in lon/lat space — straight borders like the US-Canada 49th
// parallel are straight in degrees, not great circles — which also keeps the
// rendered chords hugging the sphere.
//
// Layout (little-endian), country polylines first:
//   u32           country polyline count
//   u32           state polyline count
//   per polyline  u32 point count, then f32 (longitude, latitude) pairs
//                 in degrees
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const NATURAL_EARTH_BASE =
  "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson";
const MAX_STEP_DEGREES = 0.25;
const outPath = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data", "borders.bin");

async function downloadGeojson(layer) {
  const url = `${NATURAL_EARTH_BASE}/${layer}.geojson`;
  console.log(`Downloading ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function geometryToLines(geometry) {
  // The datasets contain a few features with null geometry.
  switch (geometry?.type) {
    case "LineString":
      return [geometry.coordinates];
    case "MultiLineString":
      return geometry.coordinates;
    default:
      return [];
  }
}

// Linear interpolation in degrees so each step spans at most MAX_STEP_DEGREES
// in both axes; sparse source vertices densify, already-dense ones pass
// through unchanged.
function subdivide(line) {
  const points = [[line[0][0], line[0][1]]];
  for (let i = 1; i < line.length; i++) {
    const [x0, y0] = line[i - 1];
    const [x1, y1] = line[i];
    const steps = Math.max(
      1,
      Math.ceil(Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) / MAX_STEP_DEGREES),
    );
    for (let s = 1; s <= steps; s++) {
      const t = s / steps;
      points.push([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]);
    }
  }
  return points;
}

function extractPolylines(geojson) {
  const polylines = [];
  for (const feature of geojson.features) {
    for (const line of geometryToLines(feature.geometry)) {
      if (line.length < 2) continue;
      polylines.push(subdivide(line));
    }
  }
  return polylines;
}

const countries = extractPolylines(await downloadGeojson("ne_10m_admin_0_boundary_lines_land"));
const states = extractPolylines(await downloadGeojson("ne_10m_admin_1_states_provinces_lines"));

const all = [...countries, ...states];
let size = 8;
for (const polyline of all) size += 4 + polyline.length * 8;

const buffer = new ArrayBuffer(size);
const view = new DataView(buffer);
let offset = 0;
view.setUint32(offset, countries.length, true);
view.setUint32(offset + 4, states.length, true);
offset += 8;
for (const polyline of all) {
  view.setUint32(offset, polyline.length, true);
  offset += 4;
  for (const [longitude, latitude] of polyline) {
    view.setFloat32(offset, longitude, true);
    view.setFloat32(offset + 4, latitude, true);
    offset += 8;
  }
}
if (offset !== size) {
  throw new Error(`layout size mismatch: wrote ${offset} of ${size}`);
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, new Uint8Array(buffer));
const points = all.reduce((sum, polyline) => sum + polyline.length, 0);
console.log(
  `Wrote ${outPath}: ${countries.length} country and ${states.length} state polylines, ` +
    `${points} points, ${(size / 1024 / 1024).toFixed(2)} MB`,
);
