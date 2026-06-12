// Geographic convention: right-handed coordinates, +Y through the north pole,
// and -Z at 90°E so that east appears screen-right when viewing the globe
// from outside with +Y up.
export function latLonToXyz(
  latitude: number,
  longitude: number,
  radius: number,
): [number, number, number] {
  return [
    radius * Math.cos(latitude) * Math.cos(longitude),
    radius * Math.sin(latitude),
    -radius * Math.cos(latitude) * Math.sin(longitude),
  ];
}

// Axis-aligned cube as a non-indexed triangle list, wound counter-clockwise
// viewed from outside (WebGPU's default front face).
export function createCube(halfExtent: number): Float32Array {
  const h = halfExtent;
  const corners: [number, number, number][] = [
    [-h, -h, -h],
    [h, -h, -h],
    [h, h, -h],
    [-h, h, -h],
    [-h, -h, h],
    [h, -h, h],
    [h, h, h],
    [-h, h, h],
  ];
  // prettier-ignore
  const indices = [
    4, 5, 6, 4, 6, 7, // +z
    1, 0, 3, 1, 3, 2, // -z
    5, 1, 2, 5, 2, 6, // +x
    0, 4, 7, 0, 7, 3, // -x
    3, 7, 6, 3, 6, 2, // +y
    0, 1, 5, 0, 5, 4, // -y
  ];
  const positions = new Float32Array(indices.length * 3);
  indices.forEach((index, i) => positions.set(corners[index], i * 3));
  return positions;
}
