// Decodes the binary produced by scripts/build-land-sdf.mjs (u32 face size,
// then 6 cubemap faces of row-major u8 signed-distance texels).
export interface LandSdf {
  faceSize: number;
  data: Uint8Array;
}

export type LandTest = (direction: readonly [number, number, number]) => boolean;

// CPU-side point test against the same cubemap the shaders sample: the
// dominant axis selects the face (inverting faceDirection in
// scripts/build-land-sdf.mjs), and a clamped bilinear read decides land
// (encoded signed distance >= 127.5). Clamping at face edges instead of
// wrapping to the neighbor is a sub-texel approximation, fine for a hover
// test.
export function createLandTest(faceSize: number, data: Uint8Array): LandTest {
  const sampleFace = (face: number, s: number, t: number): number => {
    const u = ((s + 1) / 2) * faceSize - 0.5;
    const v = ((t + 1) / 2) * faceSize - 0.5;
    const x0 = Math.max(0, Math.min(faceSize - 1, Math.floor(u)));
    const y0 = Math.max(0, Math.min(faceSize - 1, Math.floor(v)));
    const x1 = Math.min(faceSize - 1, x0 + 1);
    const y1 = Math.min(faceSize - 1, y0 + 1);
    const fx = Math.max(0, Math.min(1, u - x0));
    const fy = Math.max(0, Math.min(1, v - y0));
    const base = face * faceSize * faceSize;
    const top = data[base + y0 * faceSize + x0] * (1 - fx) + data[base + y0 * faceSize + x1] * fx;
    const bottom =
      data[base + y1 * faceSize + x0] * (1 - fx) + data[base + y1 * faceSize + x1] * fx;
    return top * (1 - fy) + bottom * fy;
  };
  return ([x, y, z]) => {
    const ax = Math.abs(x);
    const ay = Math.abs(y);
    const az = Math.abs(z);
    let face: number;
    let s: number;
    let t: number;
    if (ax >= ay && ax >= az) {
      face = x > 0 ? 0 : 1;
      s = (x > 0 ? -z : z) / ax;
      t = -y / ax;
    } else if (ay >= az) {
      face = y > 0 ? 2 : 3;
      s = x / ay;
      t = (y > 0 ? z : -z) / ay;
    } else {
      face = z > 0 ? 4 : 5;
      s = (z > 0 ? x : -x) / az;
      t = -y / az;
    }
    return sampleFace(face, s, t) >= 127.5;
  };
}

export async function loadLandSdf(url: string): Promise<LandSdf> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const faceSize = new DataView(buffer).getUint32(0, true);
  return { faceSize, data: new Uint8Array(buffer, 4) };
}
