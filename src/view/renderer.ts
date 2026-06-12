import { mat4 } from "wgpu-matrix";
import { FIELD_OF_VIEW, Model } from "../model";
import type { Borders } from "./borders";
import { Cities, visibleCityCount } from "./cities";
import { latLonToXyz } from "./geometry";
import { HIGHLIGHT_BUFFER_SIZE, INVALID_CITY, VoronoiComputer } from "./voronoi";
import linesShaderSource from "./lines.wgsl?raw";
import citiesShaderSource from "./cities.wgsl?raw";
import sphereShaderSource from "./sphere.wgsl?raw";
import blitShaderSource from "./blit.wgsl?raw";

const CITY_DOT_SIZE_PX = 4;
// Borders sit this far below the surface so on-surface geometry (Voronoi
// edges, city dots) always wins their depth test at crossings, by more than
// any chord sag. Must stay under the sphere shader's DEPTH_OFFSET or the
// globe itself would cover them.
const BORDER_SINK = 0.00075;
// The scene renders at this multiple of the canvas resolution and is then
// downsampled by the blit pass for anti-aliasing. The blit's single bilinear
// sample is only a correct box filter at exactly 2.
const SUPERSAMPLE = 2;

export class WebGPUNotSupportedError extends Error {
  constructor() {
    super("WebGPU is not supported in this browser.");
  }
}

// The visible prefix of the Lloyd-relaxed virtual site positions, read back
// from the GPU for picking; null while relaxation is off or no readback has
// landed.
export type RelaxedPositions = { data: Float32Array; count: number } | null;

interface CityBatch {
  vertexBuffer: GPUBuffer;
  // Sorted descending, so the visible set for any population threshold is a
  // prefix; see visibleCityCount.
  populations: Uint32Array;
  colorBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
}

interface BorderBatch {
  vertexBuffer: GPUBuffer;
  // Country segments occupy the instance range before the state segments.
  countrySegmentCount: number;
  stateSegmentCount: number;
  countryColorBuffer: GPUBuffer;
  stateColorBuffer: GPUBuffer;
  countryBindGroup: GPUBindGroup;
  stateBindGroup: GPUBindGroup;
}

interface SphereBatch {
  vertexBuffer: GPUBuffer;
  vertexCount: number;
  // eyeRadius vec4 plus the land, water, and highlight colors, written per
  // frame.
  paramsBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  radius: number;
}

export class Renderer {
  private depthTexture: GPUTexture;
  private colorTexture: GPUTexture;
  private canvas: HTMLCanvasElement;
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private pipeline: GPURenderPipeline;
  private citiesPipeline: GPURenderPipeline;
  private spherePipeline: GPURenderPipeline;
  private blitPipeline: GPURenderPipeline;
  private blitSampler: GPUSampler;
  private blitBindGroup: GPUBindGroup;
  private viewProjectionBuffer: GPUBuffer;
  private lineParamsBuffer: GPUBuffer;
  private citiesParamsBuffer: GPUBuffer;
  private landSdfView: GPUTextureView;
  private sdfSampler: GPUSampler;
  private highlightBuffer: GPUBuffer;
  private tooltip: HTMLElement;
  private cities: CityBatch | null = null;
  private cityInfo: Cities | null = null;
  private borders: BorderBatch | null = null;
  private sphere: SphereBatch | null = null;
  private voronoi: VoronoiComputer | null = null;
  private voronoiBindGroup: GPUBindGroup | null = null;
  private voronoiColorBuffer: GPUBuffer | null = null;
  private lastVoronoiCityCount = -1;
  private lastWeightAlpha = -1;
  private lastLloydIterations = -1;
  // Matches the zero-initialized highlight buffer (count 0, no highlight).
  private lastHovered = INVALID_CITY;
  private lastTooltipCity = INVALID_CITY;
  // Lloyd-relaxed positions read back for CPU picking. The staging buffer
  // can't be a copy target while mapped or map-pending, so a request arriving
  // mid-flight just leaves needsReadback set for a later frame.
  public pickPositions: RelaxedPositions = null;
  private relaxStaging: GPUBuffer | null = null;
  private needsReadback = false;
  private readbackPending = false;

  private constructor(
    canvas: HTMLCanvasElement,
    tooltip: HTMLElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    pipeline: GPURenderPipeline,
    citiesPipeline: GPURenderPipeline,
    spherePipeline: GPURenderPipeline,
    blitPipeline: GPURenderPipeline,
    viewProjectionBuffer: GPUBuffer,
    lineParamsBuffer: GPUBuffer,
    citiesParamsBuffer: GPUBuffer,
    highlightBuffer: GPUBuffer,
  ) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.highlightBuffer = highlightBuffer;
    this.device = device;
    this.context = context;
    this.format = format;
    this.pipeline = pipeline;
    this.citiesPipeline = citiesPipeline;
    this.spherePipeline = spherePipeline;
    this.blitPipeline = blitPipeline;
    this.viewProjectionBuffer = viewProjectionBuffer;
    this.lineParamsBuffer = lineParamsBuffer;
    this.citiesParamsBuffer = citiesParamsBuffer;
    this.depthTexture = this.createDepthTexture();
    this.colorTexture = this.createColorTexture();
    this.blitSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    this.blitBindGroup = this.createBlitBindGroup();
    this.sdfSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    // All-land 1x1 fallback: if the SDF never loads, the planet renders green
    // and Voronoi clipping degrades to a no-op instead of hiding everything.
    const fallback = device.createTexture({
      size: [1, 1, 6],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    for (let layer = 0; layer < 6; layer++) {
      device.queue.writeTexture(
        { texture: fallback, origin: [0, 0, layer] },
        new Uint8Array([255]),
        {},
        [1, 1],
      );
    }
    this.landSdfView = fallback.createView({ dimension: "cube" });
  }

  public static async create(canvas: HTMLCanvasElement, tooltip: HTMLElement): Promise<Renderer> {
    if (!navigator.gpu) {
      throw new WebGPUNotSupportedError();
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new WebGPUNotSupportedError();
    }
    const device = await adapter.requestDevice();

    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new WebGPUNotSupportedError();
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    const viewProjectionBuffer = device.createBuffer({
      size: 16 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // The hovered Voronoi cell's bounding planes, written by the highlight
    // compute pass and read by the sphere shader. Created up front (and
    // zero-initialized: count 0 = no highlight) so the sphere bind group can
    // capture it before the cities arrive.
    const highlightBuffer = device.createBuffer({
      size: HIGHLIGHT_BUFFER_SIZE,
      usage: GPUBufferUsage.STORAGE,
    });

    const module = device.createShaderModule({ code: linesShaderSource });
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vertexMain",
        buffers: [
          {
            // One segment (two vec3 endpoints) per instance; the quad corners
            // come from the vertex index.
            arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 3 * Float32Array.BYTES_PER_ELEMENT, format: "float32x3" },
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fragmentMain",
        targets: [
          {
            format,
            // The land SDF fades line fragments out across the coastline.
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const citiesModule = device.createShaderModule({ code: citiesShaderSource });
    const citiesPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: citiesModule,
        entryPoint: "vertexMain",
        buffers: [
          {
            // vec4 stride; the same buffer doubles as a compute storage
            // buffer (array<vec4f>) for the Voronoi pass later.
            arrayStride: 4 * Float32Array.BYTES_PER_ELEMENT,
            stepMode: "instance",
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
          },
        ],
      },
      fragment: {
        module: citiesModule,
        entryPoint: "fragmentMain",
        targets: [
          {
            format,
            // The same blend as the lines, so the city color's alpha works.
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const lineParamsBuffer = device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const citiesParamsBuffer = device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const sphereModule = device.createShaderModule({ code: sphereShaderSource });
    const spherePipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: sphereModule,
        entryPoint: "vertexMain",
        buffers: [
          {
            arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
          },
        ],
      },
      fragment: {
        module: sphereModule,
        entryPoint: "fragmentMain",
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            },
          },
        ],
      },
      // Rasterize the cube proxy's far faces: unlike back-face culling, this
      // keeps the sphere visible when the camera at minimum distance sits
      // inside the cube (its corners reach sqrt(3) * radius from center).
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const blitModule = device.createShaderModule({ code: blitShaderSource });
    const blitPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: blitModule, entryPoint: "vertexMain" },
      fragment: {
        module: blitModule,
        entryPoint: "fragmentMain",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });

    return new Renderer(
      canvas,
      tooltip,
      device,
      context,
      format,
      pipeline,
      citiesPipeline,
      spherePipeline,
      blitPipeline,
      viewProjectionBuffer,
      lineParamsBuffer,
      citiesParamsBuffer,
      highlightBuffer,
    );
  }

  // The signed-distance cubemap that colors the planet surface and clips
  // Voronoi edges to land. Must be set before setSphere/setCities, whose bind
  // groups capture the texture view.
  public setLandSdf(faceSize: number, data: Uint8Array): void {
    const texture = this.device.createTexture({
      size: [faceSize, faceSize, 6],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    for (let layer = 0; layer < 6; layer++) {
      this.device.queue.writeTexture(
        { texture, origin: [0, 0, layer] },
        data.subarray(layer * faceSize * faceSize, (layer + 1) * faceSize * faceSize),
        { bytesPerRow: faceSize },
        [faceSize, faceSize],
      );
    }
    this.landSdfView = texture.createView({ dimension: "cube" });
  }

  // Country and state borders drawn by the regular line pipeline at a fixed
  // thin width. Must be called after setLandSdf; the bind groups capture the
  // texture view.
  public setBorders(borders: Borders): void {
    const sunk = borders.segments.map((v) => v * (1 - BORDER_SINK));
    const vertexBuffer = this.device.createBuffer({
      size: sunk.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, sunk);

    const colorBuffer = () =>
      this.device.createBuffer({
        size: 4 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    const countryColorBuffer = colorBuffer();
    const stateColorBuffer = colorBuffer();
    const bindGroupFor = (buffer: GPUBuffer) =>
      this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.viewProjectionBuffer } },
          { binding: 1, resource: { buffer } },
          // The shared params buffer, so borders follow the line width slider.
          { binding: 2, resource: { buffer: this.lineParamsBuffer } },
          { binding: 3, resource: this.landSdfView },
          { binding: 4, resource: this.sdfSampler },
        ],
      });

    this.borders = {
      vertexBuffer,
      countrySegmentCount: borders.countrySegmentCount,
      stateSegmentCount: borders.stateSegmentCount,
      countryColorBuffer,
      stateColorBuffer,
      countryBindGroup: bindGroupFor(countryColorBuffer),
      stateBindGroup: bindGroupFor(stateColorBuffer),
    };
  }

  // The analytic sphere: positions are the cube proxy rasterized to seed the
  // per-fragment ray-sphere intersection.
  public setSphere(positions: Float32Array, radius: number): void {
    const vertexBuffer = this.device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, positions);

    const paramsBuffer = this.device.createBuffer({
      size: 16 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.spherePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewProjectionBuffer } },
        { binding: 1, resource: { buffer: paramsBuffer } },
        { binding: 2, resource: this.landSdfView },
        { binding: 3, resource: this.sdfSampler },
        { binding: 4, resource: { buffer: this.highlightBuffer } },
      ],
    });

    this.sphere = {
      vertexBuffer,
      vertexCount: positions.length / 3,
      paramsBuffer,
      bindGroup,
      radius,
    };
  }

  public setCities(cities: Cities): void {
    const { positions, populations } = cities;
    this.cityInfo = cities;
    const vertexBuffer = this.device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(vertexBuffer, 0, positions);

    const colorBuffer = this.device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.citiesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewProjectionBuffer } },
        { binding: 1, resource: { buffer: colorBuffer } },
        { binding: 2, resource: { buffer: this.citiesParamsBuffer } },
      ],
    });

    this.cities = { vertexBuffer, populations, colorBuffer, bindGroup };

    // Voronoi edges are drawn by the regular line pipeline; only the vertex
    // source (GPU-computed, indirect count) differs.
    this.voronoi = new VoronoiComputer(
      this.device,
      vertexBuffer,
      positions,
      cities.logPopulations,
      this.highlightBuffer,
    );
    this.lastVoronoiCityCount = -1;
    this.lastWeightAlpha = -1;
    this.lastLloydIterations = -1;
    this.relaxStaging = this.device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.pickPositions = null;
    const voronoiColorBuffer = this.device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.voronoiColorBuffer = voronoiColorBuffer;
    this.voronoiBindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.viewProjectionBuffer } },
        { binding: 1, resource: { buffer: voronoiColorBuffer } },
        { binding: 2, resource: { buffer: this.lineParamsBuffer } },
        { binding: 3, resource: this.landSdfView },
        { binding: 4, resource: this.sdfSampler },
      ],
    });
  }

  public render(model: Model): void {
    this.resizeIfNeeded();

    const { longitude, latitude, distance } = model.camera;
    const eye = latLonToXyz(latitude, longitude, distance);
    const view = mat4.lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const projection = mat4.perspective(
      FIELD_OF_VIEW,
      this.canvas.width / this.canvas.height,
      distance / 100,
      distance * 10,
    );
    const viewProjection = mat4.multiply(projection, view);
    this.device.queue.writeBuffer(this.viewProjectionBuffer, 0, viewProjection as Float32Array);

    const visibleCities = this.cities
      ? visibleCityCount(this.cities.populations, model.minPopulation)
      : 0;
    // An index past the visible prefix (possible briefly after a threshold
    // change) means no hover.
    const hovered =
      model.hoveredCity !== null && model.hoveredCity < visibleCities
        ? model.hoveredCity
        : INVALID_CITY;
    this.updateTooltip(hovered, viewProjection as Float32Array);

    const encoder = this.device.createCommandEncoder();
    // Rebuild the Voronoi diagram when anything that shapes the cells changes
    // (visible set, population weight, relaxation); the highlight planes
    // refresh with it. Hover alone refreshes only the highlight.
    const weightAlpha = model.weightAlpha;
    const lloydIterations = model.lloydIterations;
    if (
      this.voronoi &&
      (visibleCities !== this.lastVoronoiCityCount ||
        weightAlpha !== this.lastWeightAlpha ||
        lloydIterations !== this.lastLloydIterations)
    ) {
      this.voronoi.encode(encoder, visibleCities, hovered, weightAlpha, lloydIterations);
      this.lastVoronoiCityCount = visibleCities;
      this.lastWeightAlpha = weightAlpha;
      this.lastLloydIterations = lloydIterations;
      this.lastHovered = hovered;
      if (lloydIterations > 0 && visibleCities > 0) {
        this.needsReadback = true;
      } else {
        this.needsReadback = false;
        this.pickPositions = null;
      }
    } else if (this.voronoi && hovered !== this.lastHovered) {
      this.voronoi.encodeHighlight(encoder, visibleCities, hovered, weightAlpha);
      this.lastHovered = hovered;
    }
    let readbackCount = 0;
    if (this.voronoi && this.relaxStaging && this.needsReadback && !this.readbackPending) {
      this.voronoi.encodeReadback(encoder, this.relaxStaging, this.lastVoronoiCityCount);
      readbackCount = this.lastVoronoiCityCount;
      this.needsReadback = false;
      this.readbackPending = true;
    }
    // All colors come from the model every frame so the settings apply live.
    const colors = model.colors;
    if (this.cities) {
      this.device.queue.writeBuffer(this.cities.colorBuffer, 0, new Float32Array(colors.city));
    }
    if (this.voronoiColorBuffer) {
      this.device.queue.writeBuffer(this.voronoiColorBuffer, 0, new Float32Array(colors.voronoi));
    }
    if (this.borders) {
      this.device.queue.writeBuffer(
        this.borders.countryColorBuffer,
        0,
        new Float32Array(colors.countryBorder),
      );
      this.device.queue.writeBuffer(
        this.borders.stateColorBuffer,
        0,
        new Float32Array(colors.stateBorder),
      );
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.colorTexture.createView(),
          clearValue: {
            r: colors.background[0],
            g: colors.background[1],
            b: colors.background[2],
            a: 1,
          },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    this.device.queue.writeBuffer(
      this.lineParamsBuffer,
      0,
      new Float32Array([
        this.colorTexture.width,
        this.colorTexture.height,
        (model.lineWidth * devicePixelRatio * SUPERSAMPLE) / 2,
        0,
      ]),
    );
    // Sunk below the surface, so Voronoi edges and city dots drawn after win
    // every crossing on depth alone.
    if (this.borders && model.showBorders) {
      pass.setPipeline(this.pipeline);
      pass.setVertexBuffer(0, this.borders.vertexBuffer);
      pass.setBindGroup(0, this.borders.countryBindGroup);
      pass.draw(4, this.borders.countrySegmentCount);
      pass.setBindGroup(0, this.borders.stateBindGroup);
      pass.draw(4, this.borders.stateSegmentCount, 0, this.borders.countrySegmentCount);
    }
    if (this.voronoi && this.voronoiBindGroup) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.voronoiBindGroup);
      pass.setVertexBuffer(0, this.voronoi.edgeBuffer);
      pass.drawIndirect(this.voronoi.indirectBuffer, 0);
    }
    if (this.cities) {
      this.device.queue.writeBuffer(
        this.citiesParamsBuffer,
        0,
        new Float32Array([
          this.colorTexture.width,
          this.colorTexture.height,
          CITY_DOT_SIZE_PX * devicePixelRatio * SUPERSAMPLE,
          0,
        ]),
      );
      pass.setPipeline(this.citiesPipeline);
      pass.setBindGroup(0, this.cities.bindGroup);
      pass.setVertexBuffer(0, this.cities.vertexBuffer);
      pass.draw(4, visibleCities);
    }
    // Drawn last: surface geometry already in the depth buffer wins the depth
    // test and stays crisp, while back-side geometry is dimmed by the blend.
    if (this.sphere) {
      this.device.queue.writeBuffer(
        this.sphere.paramsBuffer,
        0,
        new Float32Array([
          eye[0],
          eye[1],
          eye[2],
          this.sphere.radius,
          ...colors.land,
          ...colors.water,
          ...colors.highlight,
        ]),
      );
      pass.setPipeline(this.spherePipeline);
      pass.setBindGroup(0, this.sphere.bindGroup);
      pass.setVertexBuffer(0, this.sphere.vertexBuffer);
      pass.draw(this.sphere.vertexCount);
    }
    pass.end();

    // Downsample the supersampled scene to the canvas.
    const blitPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    blitPass.setPipeline(this.blitPipeline);
    blitPass.setBindGroup(0, this.blitBindGroup);
    blitPass.draw(3);
    blitPass.end();

    this.device.queue.submit([encoder.finish()]);
    if (readbackCount > 0) {
      this.completeReadback(readbackCount);
    }
  }

  // Maps the staging copy of the relaxed positions and publishes it for
  // picking once it lands.
  private completeReadback(count: number): void {
    const staging = this.relaxStaging!;
    staging
      .mapAsync(GPUMapMode.READ, 0, count * 16)
      .then(() => {
        const data = new Float32Array(staging.getMappedRange(0, count * 16)).slice();
        staging.unmap();
        // Relaxation may have been switched off while the map was in flight;
        // stale-but-superseded data is dropped instead of resurrected.
        if (this.lastLloydIterations > 0) {
          this.pickPositions = { data, count };
        }
        this.readbackPending = false;
      })
      .catch(() => {
        this.readbackPending = false;
      });
  }

  // Anchors the tooltip to the hovered city's projected screen position and
  // rebuilds its label when the hovered city changes.
  private updateTooltip(hovered: number, viewProjection: Float32Array): void {
    if (hovered === INVALID_CITY || !this.cityInfo) {
      this.tooltip.hidden = true;
      this.lastTooltipCity = INVALID_CITY;
      return;
    }
    const positions = this.cityInfo.positions;
    const x = positions[hovered * 4];
    const y = positions[hovered * 4 + 1];
    const z = positions[hovered * 4 + 2];
    const m = viewProjection; // column-major
    const clipX = m[0] * x + m[4] * y + m[8] * z + m[12];
    const clipY = m[1] * x + m[5] * y + m[9] * z + m[13];
    const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];
    this.tooltip.style.left = `${((clipX / clipW) * 0.5 + 0.5) * this.canvas.clientWidth}px`;
    this.tooltip.style.top = `${(0.5 - (clipY / clipW) * 0.5) * this.canvas.clientHeight}px`;
    if (hovered !== this.lastTooltipCity) {
      this.lastTooltipCity = hovered;
      const label = this.cityInfo.label(hovered);
      this.tooltip.replaceChildren();
      for (const [className, text] of [
        ["title", label.title],
        ["detail", label.region],
        ["detail", label.population],
        ["badge", label.badge],
      ]) {
        if (text === "") continue;
        const line = document.createElement("div");
        line.className = className;
        line.textContent = text;
        this.tooltip.append(line);
      }
    }
    this.tooltip.hidden = false;
  }

  private resizeIfNeeded(): void {
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * devicePixelRatio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * devicePixelRatio));
    if (this.canvas.width === width && this.canvas.height === height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    this.depthTexture.destroy();
    this.depthTexture = this.createDepthTexture();
    this.colorTexture.destroy();
    this.colorTexture = this.createColorTexture();
    this.blitBindGroup = this.createBlitBindGroup();
  }

  private createDepthTexture(): GPUTexture {
    return this.device.createTexture({
      size: [
        Math.max(1, this.canvas.width) * SUPERSAMPLE,
        Math.max(1, this.canvas.height) * SUPERSAMPLE,
      ],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private createColorTexture(): GPUTexture {
    return this.device.createTexture({
      size: [
        Math.max(1, this.canvas.width) * SUPERSAMPLE,
        Math.max(1, this.canvas.height) * SUPERSAMPLE,
      ],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  private createBlitBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.blitSampler },
        { binding: 1, resource: this.colorTexture.createView() },
      ],
    });
  }
}
