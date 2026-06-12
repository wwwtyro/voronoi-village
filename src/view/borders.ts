import { latLonToXyz } from "./geometry";

// Decodes the binary produced by scripts/build-borders.mjs (country then
// state polylines of lon/lat degree pairs) and expands the polylines into the
// line pipeline's instance layout: two unit-sphere vec3 endpoints per
// segment, country segments first so each class is a contiguous range.
export interface Borders {
  segments: Float32Array;
  countrySegmentCount: number;
  stateSegmentCount: number;
}

export async function loadBorders(url: string): Promise<Borders> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const countryCount = view.getUint32(0, true);
  const stateCount = view.getUint32(4, true);

  // An n-point polyline contributes n - 1 segments.
  let offset = 8;
  let segmentCount = 0;
  for (let i = 0; i < countryCount + stateCount; i++) {
    const points = view.getUint32(offset, true);
    segmentCount += points - 1;
    offset += 4 + points * 8;
  }

  const segments = new Float32Array(segmentCount * 6);
  offset = 8;
  let out = 0;
  let countrySegmentCount = 0;
  for (let i = 0; i < countryCount + stateCount; i++) {
    const points = view.getUint32(offset, true);
    offset += 4;
    let previous: [number, number, number] | null = null;
    for (let p = 0; p < points; p++) {
      const longitude = (view.getFloat32(offset, true) * Math.PI) / 180;
      const latitude = (view.getFloat32(offset + 4, true) * Math.PI) / 180;
      offset += 8;
      const position = latLonToXyz(latitude, longitude, 1);
      if (previous) {
        segments.set(previous, out);
        segments.set(position, out + 3);
        out += 6;
      }
      previous = position;
    }
    if (i === countryCount - 1) {
      countrySegmentCount = out / 6;
    }
  }
  return {
    segments,
    countrySegmentCount,
    stateSegmentCount: segmentCount - countrySegmentCount,
  };
}
