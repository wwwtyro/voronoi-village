import { latLonToXyz } from "./geometry";

// Indexed by the capital-type byte in cities.bin.
const CAPITAL_LABELS = ["", "National capital", "State capital", "County seat"];
const NO_ADMIN1 = 0xffff;

const textDecoder = new TextDecoder();
const regionNames = new Intl.DisplayNames("en", { type: "region" });

export interface CityLabel {
  title: string;
  // "Admin1, Country", either part omitted when unknown.
  region: string;
  // "Pop. 1,234,567", or "" when the source data has no population.
  population: string;
  // Capital status, or "" for ordinary cities.
  badge: string;
}

export class Cities {
  // xyz unit vectors padded to a 16-byte stride (4 floats per city) so the
  // same GPU buffer can serve as an instance buffer and a compute pass
  // storage buffer (array<vec4f>).
  public readonly positions: Float32Array;
  // Parallel to positions, sorted descending; any minimum-population filter
  // is a prefix of the arrays.
  public readonly populations: Uint32Array;
  // log10(population / 500) per city, floored at 0: the population-weighted
  // (power) Voronoi diagram's per-site weight before the alpha scale, shared
  // by the GPU compute pass and CPU picking.
  public readonly logPopulations: Float32Array;
  private types: Uint8Array;
  private countryCodes: Uint8Array;
  private admin1Index: Uint16Array;
  private admin1Names: string[];
  private nameBytes: Uint8Array;
  private nameStart: Uint32Array;
  private nameLength: Uint16Array;

  constructor(args: {
    positions: Float32Array;
    populations: Uint32Array;
    types: Uint8Array;
    countryCodes: Uint8Array;
    admin1Index: Uint16Array;
    admin1Names: string[];
    nameBytes: Uint8Array;
    nameStart: Uint32Array;
    nameLength: Uint16Array;
  }) {
    this.positions = args.positions;
    this.populations = args.populations;
    this.logPopulations = new Float32Array(args.populations.length);
    for (let i = 0; i < args.populations.length; i++) {
      this.logPopulations[i] = Math.log10(Math.max(args.populations[i], 500) / 500);
    }
    this.types = args.types;
    this.countryCodes = args.countryCodes;
    this.admin1Index = args.admin1Index;
    this.admin1Names = args.admin1Names;
    this.nameBytes = args.nameBytes;
    this.nameStart = args.nameStart;
    this.nameLength = args.nameLength;
  }

  // Decoded lazily: eagerly decoding ~200k strings is noticeable at startup
  // while a hover needs only one.
  public name(index: number): string {
    const start = this.nameStart[index];
    return textDecoder.decode(this.nameBytes.subarray(start, start + this.nameLength[index]));
  }

  public label(index: number): CityLabel {
    const admin1 =
      this.admin1Index[index] === NO_ADMIN1 ? "" : this.admin1Names[this.admin1Index[index]];
    const code = String.fromCharCode(
      this.countryCodes[index * 2],
      this.countryCodes[index * 2 + 1],
    ).trim();
    let country = code;
    try {
      country = regionNames.of(code) ?? code;
    } catch {
      // Not a valid region code; show it raw.
    }
    return {
      title: this.name(index),
      region: [admin1, country].filter(Boolean).join(", "),
      population:
        this.populations[index] === 0
          ? ""
          : `Pop. ${this.populations[index].toLocaleString("en-US")}`,
      badge: CAPITAL_LABELS[this.types[index]] ?? "",
    };
  }
}

export async function loadCities(url: string): Promise<Cities> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  const cityCount = view.getUint32(0, true);
  let offset = 4;
  const lonLat = new Float32Array(buffer, offset, cityCount * 2);
  offset += cityCount * 8;
  const populations = new Uint32Array(buffer, offset, cityCount).slice();
  offset += cityCount * 4;
  const types = bytes.subarray(offset, offset + cityCount);
  offset += cityCount;
  const countryCodes = bytes.subarray(offset, offset + cityCount * 2);
  offset += cityCount * 2;
  // Read through the DataView: this offset is odd when the count is.
  const admin1Index = new Uint16Array(cityCount);
  for (let i = 0; i < cityCount; i++) {
    admin1Index[i] = view.getUint16(offset, true);
    offset += 2;
  }
  const admin1Count = view.getUint32(offset, true);
  offset += 4;
  const admin1Names: string[] = [];
  for (let i = 0; i < admin1Count; i++) {
    const length = view.getUint16(offset, true);
    offset += 2;
    admin1Names.push(textDecoder.decode(bytes.subarray(offset, offset + length)));
    offset += length;
  }
  const nameStart = new Uint32Array(cityCount);
  const nameLength = new Uint16Array(cityCount);
  for (let i = 0; i < cityCount; i++) {
    nameLength[i] = view.getUint16(offset, true);
    offset += 2;
    nameStart[i] = offset;
    offset += nameLength[i];
  }

  const positions = new Float32Array(cityCount * 4);
  for (let i = 0; i < cityCount; i++) {
    const longitude = (lonLat[i * 2] * Math.PI) / 180;
    const latitude = (lonLat[i * 2 + 1] * Math.PI) / 180;
    const [x, y, z] = latLonToXyz(latitude, longitude, 1);
    positions[i * 4] = x;
    positions[i * 4 + 1] = y;
    positions[i * 4 + 2] = z;
  }

  return new Cities({
    positions,
    populations,
    types,
    countryCodes,
    admin1Index,
    admin1Names,
    nameBytes: bytes,
    nameStart,
    nameLength,
  });
}

// Cities are sorted by population descending, so the visible set for any
// threshold is a prefix; binary search for the first city below it.
export function visibleCityCount(populations: Uint32Array, minPopulation: number): number {
  let low = 0;
  let high = populations.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (populations[mid] >= minPopulation) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}
