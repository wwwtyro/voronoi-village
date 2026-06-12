// Downloads the GeoNames cities500 dump (settlements with population >= 500)
// plus the admin1 names table and flattens them into a compact binary for the
// app. Cities are sorted by population descending so any minimum-population
// filter is a prefix of the arrays. Layout (little-endian):
//   u32              city count
//   f32 pairs        (longitude, latitude) in degrees, all cities
//   u32 per city     population
//   u8 per city      capital type (0 none, 1 national, 2 admin1, 3 admin2)
//   u8 x 2 per city  ISO 3166 country code, ASCII
//   u16 per city     index into the admin1 name table (0xffff = none)
//   u32              admin1 name count
//   per admin1 name  u16 UTF-8 byte length, then the bytes
//   per city         u16 UTF-8 byte length, then the name bytes
import { unzipSync } from "fflate";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DUMP_URL = "https://download.geonames.org/export/dump/cities500.zip";
const ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt";
const MIN_POPULATION = 500; // the dump also contains population-0 admin seats
const CAPITAL_TYPES = { PPLC: 1, PPLA: 2, PPLA2: 3 };
const NO_ADMIN1 = 0xffff;
const outPath = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "data", "cities.bin");

async function download(url) {
  console.log(`Downloading ${url}`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

const text = new TextDecoder().decode(unzipSync(await download(DUMP_URL))["cities500.txt"]);

// "CC.A1" -> admin1 name, e.g. "US.TX" -> "Texas".
const admin1Lookup = new Map();
for (const line of new TextDecoder().decode(await download(ADMIN1_URL)).split("\n")) {
  if (line.length === 0) continue;
  const columns = line.split("\t");
  admin1Lookup.set(columns[0], columns[1]);
}

// GeoNames "geoname" table columns: 1 = name, 4 = latitude, 5 = longitude,
// 7 = feature code, 8 = country code, 10 = admin1 code, 14 = population.
const cities = [];
for (const line of text.split("\n")) {
  if (line.length === 0) continue;
  const columns = line.split("\t");
  const population = Number(columns[14]);
  if (population < MIN_POPULATION) continue;
  cities.push({
    name: columns[1],
    longitude: Number(columns[5]),
    latitude: Number(columns[4]),
    population,
    type: CAPITAL_TYPES[columns[7]] ?? 0,
    country: columns[8],
    admin1: admin1Lookup.get(`${columns[8]}.${columns[10]}`),
  });
}
cities.sort((a, b) => b.population - a.population);

// Dedupe the admin1 names actually used into an indexed table.
const admin1Index = new Map();
for (const city of cities) {
  if (city.admin1 !== undefined && !admin1Index.has(city.admin1)) {
    admin1Index.set(city.admin1, admin1Index.size);
  }
}
if (admin1Index.size >= NO_ADMIN1) {
  throw new Error("admin1 name table overflows u16 indices");
}

const encoder = new TextEncoder();
const admin1Bytes = [...admin1Index.keys()].map((name) => encoder.encode(name));
const nameBytes = cities.map((city) => encoder.encode(city.name));

const n = cities.length;
let size = 4 + n * 8 + n * 4 + n + n * 2 + n * 2 + 4;
for (const b of admin1Bytes) size += 2 + b.length;
for (const b of nameBytes) size += 2 + b.length;

const buffer = new ArrayBuffer(size);
const view = new DataView(buffer);
const bytes = new Uint8Array(buffer);
let offset = 0;
view.setUint32(offset, n, true);
offset += 4;
for (const city of cities) {
  view.setFloat32(offset, city.longitude, true);
  view.setFloat32(offset + 4, city.latitude, true);
  offset += 8;
}
for (const city of cities) {
  view.setUint32(offset, city.population, true);
  offset += 4;
}
for (const city of cities) {
  bytes[offset++] = city.type;
}
for (const city of cities) {
  // NaN from charCodeAt past the end falls back to a space.
  bytes[offset++] = city.country.charCodeAt(0) || 32;
  bytes[offset++] = city.country.charCodeAt(1) || 32;
}
for (const city of cities) {
  view.setUint16(offset, city.admin1 !== undefined ? admin1Index.get(city.admin1) : NO_ADMIN1, true);
  offset += 2;
}
view.setUint32(offset, admin1Bytes.length, true);
offset += 4;
for (const b of [...admin1Bytes, ...nameBytes]) {
  view.setUint16(offset, b.length, true);
  offset += 2;
  bytes.set(b, offset);
  offset += b.length;
}
if (offset !== size) {
  throw new Error(`layout size mismatch: wrote ${offset} of ${size}`);
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, bytes);
console.log(
  `Wrote ${outPath}: ${n} cities, ${admin1Bytes.length} admin1 names, ${(size / 1024 / 1024).toFixed(2)} MB`,
);
