import shaderSource from "./voronoi.wgsl?raw";

const ROWS = 128; // keep in sync with voronoi.wgsl
const WORKGROUP_SIZE = 64; // keep in sync with voronoi.wgsl
// Line-list vertex budget for the edge output buffer (~24 MB). Worst case
// (all ~195k cities visible) needs roughly 1.5M vertices.
const VERTEX_CAPACITY = 2_097_152;

// Hovered-city sentinel: any index >= the visible count means "no cell", and
// highlightMain publishes an empty plane list for it.
export const INVALID_CITY = 0xffffffff;
// u32 count (padded to 16 bytes) + 33 vec4f planes; keep in sync with the
// HighlightCell struct in voronoi.wgsl and sphere.wgsl.
export const HIGHLIGHT_BUFFER_SIZE = 16 + 33 * 16;
// Angular radius of the initial cell cap; cells are truncated to it. Keep in
// sync with voronoi.wgsl.
export const CAP_RADIUS = 1.3962634;
// Maximum relaxed-site displacement, radians; keep in sync with voronoi.wgsl.
const S_MAX = 0.05;

// Computes spherical Voronoi cell edges for the K most populous cities on the
// GPU. The spatial grid is built once over all cities; visibility at a given
// population threshold is just the index test `city < K` in the shader, so a
// threshold change only requires a re-dispatch with a new K.
export class VoronoiComputer {
  public readonly edgeBuffer: GPUBuffer;
  public readonly indirectBuffer: GPUBuffer;
  private device: GPUDevice;
  private mainPipeline: GPUComputePipeline;
  private finalizePipeline: GPUComputePipeline;
  private highlightPipeline: GPUComputePipeline;
  private relaxPipeline: GPUComputePipeline;
  // Indexed by position source: 0 = true positions, 1 = relaxed A, 2 = relaxed B.
  private mainBindGroups: GPUBindGroup[];
  private highlightBindGroups: GPUBindGroup[];
  // Relax steps ping-pong: [0] orig -> A, [1] A -> B, [2] B -> A.
  private relaxBindGroups: GPUBindGroup[];
  private finalizeBindGroup: GPUBindGroup;
  private uniformBuffer: GPUBuffer;
  private counterBuffer: GPUBuffer;
  private relaxedBuffers: [GPUBuffer, GPUBuffer];
  // Which position source the last encode left the diagram built from; the
  // hover-only highlight refresh and the picking readback must match it.
  private finalSource = 0;
  private lastSlack = 0;

  constructor(
    device: GPUDevice,
    positionsBuffer: GPUBuffer,
    positions: Float32Array,
    logPopulations: Float32Array,
    highlightBuffer: GPUBuffer,
  ) {
    this.device = device;

    const { rowStart, binOffsets, binEntries } = buildBins(positions);
    const rowStartBuffer = createStorageBuffer(device, rowStart);
    const binOffsetsBuffer = createStorageBuffer(device, binOffsets);
    const binEntriesBuffer = createStorageBuffer(device, binEntries);
    const logPopBuffer = createStorageBuffer(device, logPopulations);

    this.edgeBuffer = device.createBuffer({
      size: VERTEX_CAPACITY * 3 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    });
    this.counterBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.indirectBuffer = device.createBuffer({
      size: 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    this.uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Lloyd-relaxed virtual site positions; COPY_SRC for the picking readback.
    const createRelaxedBuffer = () =>
      device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
    this.relaxedBuffers = [createRelaxedBuffer(), createRelaxedBuffer()];

    const module = device.createShaderModule({ code: shaderSource });
    this.mainPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "voronoiMain" },
    });
    this.finalizePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "finalizeMain" },
    });
    this.highlightPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "highlightMain" },
    });
    this.relaxPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "relaxMain" },
    });

    const positionSources = [positionsBuffer, this.relaxedBuffers[0], this.relaxedBuffers[1]];
    this.mainBindGroups = positionSources.map((source) =>
      device.createBindGroup({
        layout: this.mainPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: source } },
          { binding: 2, resource: { buffer: rowStartBuffer } },
          { binding: 3, resource: { buffer: binOffsetsBuffer } },
          { binding: 4, resource: { buffer: binEntriesBuffer } },
          { binding: 5, resource: { buffer: this.edgeBuffer } },
          { binding: 6, resource: { buffer: this.counterBuffer } },
          { binding: 9, resource: { buffer: logPopBuffer } },
        ],
      }),
    );
    this.highlightBindGroups = positionSources.map((source) =>
      device.createBindGroup({
        layout: this.highlightPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: source } },
          { binding: 2, resource: { buffer: rowStartBuffer } },
          { binding: 3, resource: { buffer: binOffsetsBuffer } },
          { binding: 4, resource: { buffer: binEntriesBuffer } },
          { binding: 8, resource: { buffer: highlightBuffer } },
          { binding: 9, resource: { buffer: logPopBuffer } },
        ],
      }),
    );
    this.relaxBindGroups = [
      [positionsBuffer, this.relaxedBuffers[0]],
      [this.relaxedBuffers[0], this.relaxedBuffers[1]],
      [this.relaxedBuffers[1], this.relaxedBuffers[0]],
    ].map(([input, output]) =>
      device.createBindGroup({
        layout: this.relaxPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: input } },
          { binding: 2, resource: { buffer: rowStartBuffer } },
          { binding: 3, resource: { buffer: binOffsetsBuffer } },
          { binding: 4, resource: { buffer: binEntriesBuffer } },
          { binding: 9, resource: { buffer: logPopBuffer } },
          { binding: 10, resource: { buffer: output } },
          { binding: 11, resource: { buffer: positionsBuffer } },
        ],
      }),
    );
    this.finalizeBindGroup = device.createBindGroup({
      layout: this.finalizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 6, resource: { buffer: this.counterBuffer } },
        { binding: 7, resource: { buffer: this.indirectBuffer } },
      ],
    });
  }

  // Rebuild everything: relax the virtual sites from scratch (deterministic,
  // no accumulation across runs), then the cell edges and the hovered cell's
  // highlight planes from the final positions.
  public encode(
    encoder: GPUCommandEncoder,
    visibleCount: number,
    hovered: number,
    alpha: number,
    iterations: number,
  ): void {
    this.lastSlack = iterations > 0 ? 2 * S_MAX : 0;
    this.writeUniforms(visibleCount, hovered, alpha);
    this.device.queue.writeBuffer(this.counterBuffer, 0, new Uint32Array([0]));
    const workgroups = Math.ceil(visibleCount / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.relaxPipeline);
    for (let step = 0; step < iterations; step++) {
      pass.setBindGroup(0, this.relaxBindGroups[step === 0 ? 0 : 2 - (step % 2)]);
      pass.dispatchWorkgroups(workgroups);
    }
    this.finalSource = iterations === 0 ? 0 : 2 - (iterations % 2);
    pass.setPipeline(this.mainPipeline);
    pass.setBindGroup(0, this.mainBindGroups[this.finalSource]);
    pass.dispatchWorkgroups(workgroups);
    pass.setPipeline(this.finalizePipeline);
    pass.setBindGroup(0, this.finalizeBindGroup);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(this.highlightPipeline);
    pass.setBindGroup(0, this.highlightBindGroups[this.finalSource]);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  // Refresh only the hovered cell's highlight planes; the edges are untouched.
  public encodeHighlight(
    encoder: GPUCommandEncoder,
    visibleCount: number,
    hovered: number,
    alpha: number,
  ): void {
    this.writeUniforms(visibleCount, hovered, alpha);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.highlightPipeline);
    pass.setBindGroup(0, this.highlightBindGroups[this.finalSource]);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  // Copy the visible prefix of the relaxed positions for CPU picking. Only
  // meaningful after an encode with iterations > 0.
  public encodeReadback(encoder: GPUCommandEncoder, staging: GPUBuffer, visibleCount: number): void {
    if (this.finalSource === 0 || visibleCount === 0) {
      return;
    }
    encoder.copyBufferToBuffer(this.relaxedBuffers[this.finalSource - 1], 0, staging, 0, visibleCount * 16);
  }

  // A single shared write: encode and encodeHighlight may both run before one
  // submit, and two writes with different contents would clobber each other.
  private writeUniforms(visibleCount: number, hovered: number, alpha: number): void {
    const data = new ArrayBuffer(32);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    u32[0] = visibleCount;
    u32[1] = VERTEX_CAPACITY;
    u32[2] = hovered;
    f32[3] = alpha;
    f32[4] = this.lastSlack;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }
}

// Lat/lon grid with ROWS latitude rows and a per-row column count
// proportional to cos(latitude), for roughly equal-area bins. Cities land in
// bins in ascending index order (= descending population), which lets the
// shader stop scanning a bin at the first index past K.
function buildBins(positions: Float32Array): {
  rowStart: Uint32Array;
  binOffsets: Uint32Array;
  binEntries: Uint32Array;
} {
  const cityCount = positions.length / 4;
  const dLat = Math.PI / ROWS;

  const rowStart = new Uint32Array(ROWS + 1);
  for (let row = 0; row < ROWS; row++) {
    const centerLat = (row + 0.5) * dLat - Math.PI / 2;
    rowStart[row + 1] = rowStart[row] + Math.max(1, Math.round(2 * ROWS * Math.cos(centerLat)));
  }
  const totalBins = rowStart[ROWS];

  const bins = new Uint32Array(cityCount);
  for (let i = 0; i < cityCount; i++) {
    const latitude = Math.asin(Math.min(1, Math.max(-1, positions[i * 4 + 1])));
    const longitude = Math.atan2(-positions[i * 4 + 2], positions[i * 4]);
    const row = Math.min(ROWS - 1, Math.max(0, Math.floor((latitude + Math.PI / 2) / dLat)));
    const colCount = rowStart[row + 1] - rowStart[row];
    const col = Math.min(
      colCount - 1,
      Math.max(0, Math.floor(((longitude + Math.PI) / (2 * Math.PI)) * colCount)),
    );
    bins[i] = rowStart[row] + col;
  }

  const binOffsets = new Uint32Array(totalBins + 1);
  for (let i = 0; i < cityCount; i++) {
    binOffsets[bins[i] + 1]++;
  }
  for (let bin = 0; bin < totalBins; bin++) {
    binOffsets[bin + 1] += binOffsets[bin];
  }
  const binEntries = new Uint32Array(cityCount);
  const cursor = binOffsets.slice(0, totalBins);
  for (let i = 0; i < cityCount; i++) {
    binEntries[cursor[bins[i]]++] = i;
  }

  return { rowStart, binOffsets, binEntries };
}

function createStorageBuffer(device: GPUDevice, data: Uint32Array | Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
