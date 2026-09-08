import {
  buildLayout,
  initParams,
  ENC_CODE,
  type ModelLayout,
  type ModelSpec,
} from './spec';
import type { HyperParams, NeuronPreview, Trainer, ViewBox } from './trainer';
import { BLIT_WGSL, RENDER_WGSL, TRAIN_WGSL } from './wgsl';

/**
 * `Float32Array` is generic over its backing buffer in TS 5.7+, while the
 * WebGPU types insist on a non-shared `ArrayBuffer`. Going through the raw
 * buffer keeps the call sites clean.
 */
function writeF32(queue: GPUQueue, target: GPUBuffer, data: Float32Array, elements?: number) {
  const count = elements ?? data.length;
  queue.writeBuffer(target, 0, data.buffer as ArrayBuffer, data.byteOffset, count * 4);
}

const UNIFORM_STRIDE = 256; // minUniformBufferOffsetAlignment
const MAX_STEPS_PER_FRAME = 64;
const MAX_IMAGE_DIM = 512;
const LOSS_WORKGROUP = 64;
const MAX_LOSS_PARTIALS = 512;

export async function requestDevice(): Promise<GPUDevice | null> {
  if (!('gpu' in navigator)) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    return await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize,
          512 * 1024 * 1024,
        ),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 512 * 1024 * 1024),
      },
    });
  } catch {
    return null;
  }
}

/**
 * A ping-pong pair of activation buffers plus a packed rgba output, used to
 * evaluate the field over a regular grid (the visible canvas, and the smaller
 * grid behind the per-neuron thumbnails).
 */
class RenderTarget {
  readonly bufA: GPUBuffer;
  readonly bufB: GPUBuffer;
  readonly outBuf: GPUBuffer;
  readonly uniform: GPUBuffer;
  bind0!: GPUBindGroup;
  bindAB!: GPUBindGroup;
  bindBA!: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    readonly width: number,
    readonly height: number,
    readonly stride: number,
  ) {
    const cells = width * height * stride * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    this.bufA = device.createBuffer({ size: cells, usage });
    this.bufB = device.createBuffer({ size: cells, usage });
    this.outBuf = device.createBuffer({
      size: width * height * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.uniform = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  writeUniform(layout: ModelLayout, view: ViewBox) {
    const buf = new ArrayBuffer(48);
    const dv = new DataView(buf);
    dv.setUint32(0, this.width, true);
    dv.setUint32(4, this.height, true);
    dv.setUint32(8, layout.inputDim, true);
    dv.setUint32(12, ENC_CODE[layout.spec.encoding], true);
    dv.setUint32(16, layout.spec.frequencies, true);
    dv.setUint32(20, this.stride, true);
    dv.setFloat32(32, view.zoom, true);
    dv.setFloat32(36, view.cx, true);
    dv.setFloat32(40, view.cy, true);
    this.device.queue.writeBuffer(this.uniform, 0, buf);
  }

  /** Buffer that layer `i` writes into. rEncode writes A, so layers alternate B, A, B… */
  bufferAfterLayer(i: number): GPUBuffer {
    return i % 2 === 0 ? this.bufB : this.bufA;
  }

  /** Bind group whose `src` is the buffer produced by layer `i`. */
  bindReadingLayer(i: number): GPUBindGroup {
    return i % 2 === 0 ? this.bindBA : this.bindAB;
  }

  destroy() {
    this.bufA.destroy();
    this.bufB.destroy();
    this.outBuf.destroy();
    this.uniform.destroy();
  }
}

export class GpuTrainer implements Trainer {
  readonly backend = 'webgpu' as const;

  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;

  // Persistent device resources
  private params!: GPUBuffer;
  private moments!: GPUBuffer;
  private acts!: GPUBuffer;
  private pre!: GPUBuffer;
  private deltas!: GPUBuffer;
  private image!: GPUBuffer;
  private aux!: GPUBuffer;
  private encBuf!: GPUBuffer;
  private lossStage!: GPUBuffer;
  private paramStage!: GPUBuffer;
  private previewStage!: GPUBuffer;
  private globalsBuf!: GPUBuffer;
  private layerBuf!: GPUBuffer;
  private blitUniform!: GPUBuffer;

  private trainBGL0!: GPUBindGroupLayout;
  private trainBGL1!: GPUBindGroupLayout;
  private trainBind0!: GPUBindGroup;
  private trainBind1!: GPUBindGroup;

  private renderBGL0!: GPUBindGroupLayout;
  private renderBGL1!: GPUBindGroupLayout;
  private renderBGL2!: GPUBindGroupLayout;
  private renderBind2!: GPUBindGroup;

  private pEncode!: GPUComputePipeline;
  private pForward!: GPUComputePipeline;
  private pOutputGrad!: GPUComputePipeline;
  private pBackward!: GPUComputePipeline;
  private pWeightGradAdam!: GPUComputePipeline;

  private pREncode!: GPUComputePipeline;
  private pRForward!: GPUComputePipeline;
  private pRResolve!: GPUComputePipeline;
  private pBlit!: GPURenderPipeline;
  private blitBGL!: GPUBindGroupLayout;
  private blitBind!: GPUBindGroup;

  private main!: RenderTarget;
  private preview!: RenderTarget;

  private hyper: HyperParams;
  private view: ViewBox = { zoom: 1, cx: 0, cy: 0 };
  private cpuParams: Float32Array;
  private imgW = 1;
  private imgH = 1;
  private allocatedBatch = 0;
  private actOffsets: { inOff: number[]; preOff: number[]; postOff: number[] } = {
    inOff: [],
    preOff: [],
    postOff: [],
  };

  private lastLoss = 0;
  private lossPending = false;
  private lossPartials = 1;
  private paramReadPending = false;
  private previewPending = false;
  private destroyed = false;

  stepCount = 0;

  private constructor(
    private device: GPUDevice,
    readonly canvas: HTMLCanvasElement,
    readonly layout: ModelLayout,
    hyper: HyperParams,
    outW: number,
    outH: number,
  ) {
    this.hyper = { ...hyper };
    this.cpuParams = initParams(layout);

    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('could not acquire a webgpu canvas context');
    this.ctx = ctx;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.createPipelines();
    this.createStaticBuffers();
    this.allocBatchBuffers(this.hyper.batch);
    this.setOutputSize(outW, outH);
    this.uploadParams();
  }

  static create(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    spec: ModelSpec,
    hyper: HyperParams,
    outW: number,
    outH: number,
  ): GpuTrainer {
    return new GpuTrainer(device, canvas, buildLayout(spec), hyper, outW, outH);
  }

  // ---------------------------------------------------------------- setup

  private createPipelines() {
    const d = this.device;
    const st = (type: GPUBufferBindingType, binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });

    // Exactly 8 storage-type bindings (0-7) — the guaranteed minimum for
    // maxStorageBuffersPerShaderStage on any conformant WebGPU device. See the
    // comment above TRAIN_WGSL for how these map to the trainer's arrays.
    this.trainBGL0 = d.createBindGroupLayout({
      entries: [
        st('storage', 0),
        st('storage', 1),
        st('storage', 2),
        st('storage', 3),
        st('storage', 4),
        st('read-only-storage', 5),
        st('storage', 6),
        st('read-only-storage', 7),
        {
          binding: 8,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 },
        },
      ],
    });
    this.trainBGL1 = d.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 },
        },
      ],
    });

    const trainLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.trainBGL0, this.trainBGL1],
    });
    const trainModule = d.createShaderModule({ code: TRAIN_WGSL, label: 'train' });
    const cp = (entryPoint: string) =>
      d.createComputePipeline({ layout: trainLayout, compute: { module: trainModule, entryPoint } });

    this.pEncode = cp('encode');
    this.pForward = cp('forward');
    this.pOutputGrad = cp('outputGrad');
    this.pBackward = cp('backwardDelta');
    this.pWeightGradAdam = cp('weightGradAdam');

    this.renderBGL0 = d.createBindGroupLayout({
      entries: [
        st('read-only-storage', 0),
        st('read-only-storage', 1),
        st('storage', 2),
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.renderBGL1 = d.createBindGroupLayout({
      entries: [st('read-only-storage', 0), st('storage', 1)],
    });
    this.renderBGL2 = d.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 64 },
        },
      ],
    });

    const renderLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.renderBGL0, this.renderBGL1, this.renderBGL2],
    });
    const renderModule = d.createShaderModule({ code: RENDER_WGSL, label: 'render' });
    const rp = (entryPoint: string) =>
      d.createComputePipeline({
        layout: renderLayout,
        compute: { module: renderModule, entryPoint },
      });
    this.pREncode = rp('rEncode');
    this.pRForward = rp('rForward');
    this.pRResolve = rp('rResolve');

    this.blitBGL = d.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'read-only-storage' },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const blitModule = d.createShaderModule({ code: BLIT_WGSL, label: 'blit' });
    this.pBlit = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.blitBGL] }),
      vertex: { module: blitModule, entryPoint: 'vs' },
      fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  private createStaticBuffers() {
    const d = this.device;
    const n = this.layout.paramCount;
    const S = GPUBufferUsage.STORAGE;

    this.params = d.createBuffer({
      size: n * 4,
      usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // mom1 occupies [0, n), mom2 occupies [n, 2n) — one binding instead of two.
    this.moments = d.createBuffer({ size: n * 2 * 4, usage: S | GPUBufferUsage.COPY_DST });

    this.image = d.createBuffer({
      size: MAX_IMAGE_DIM * MAX_IMAGE_DIM * 16,
      usage: S | GPUBufferUsage.COPY_DST,
    });

    // WebGPU forbids zero-sized buffers, and `none` encoding has no constants.
    const encBytes = Math.max(this.layout.encData.byteLength, 4);
    this.encBuf = d.createBuffer({ size: encBytes, usage: S | GPUBufferUsage.COPY_DST });
    if (this.layout.encData.length > 0) {
      writeF32(d.queue, this.encBuf, this.layout.encData);
    }

    this.lossStage = d.createBuffer({
      size: MAX_LOSS_PARTIALS * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.paramStage = d.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    this.globalsBuf = d.createBuffer({
      size: UNIFORM_STRIDE * MAX_STEPS_PER_FRAME,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.layerBuf = d.createBuffer({
      size: UNIFORM_STRIDE * Math.max(this.layout.layers.length, 1),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.blitUniform = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.trainBind1 = d.createBindGroup({
      layout: this.trainBGL1,
      entries: [{ binding: 0, resource: { buffer: this.layerBuf, size: 64 } }],
    });
    this.renderBind2 = d.createBindGroup({
      layout: this.renderBGL2,
      entries: [{ binding: 0, resource: { buffer: this.layerBuf, size: 64 } }],
    });
  }

  /**
   * Activation/gradient scratch scales with the batch, so it is reallocated
   * whenever the batch size changes.
   */
  private allocBatchBuffers(batch: number) {
    if (batch === this.allocatedBatch) return;
    this.acts?.destroy();
    this.pre?.destroy();
    this.deltas?.destroy();
    this.aux?.destroy();

    const inOff: number[] = [];
    const preOff: number[] = [];
    const postOff: number[] = [];
    let actCursor = batch * this.layout.inputDim; // features occupy the head of `acts`
    let preCursor = 0;
    for (const l of this.layout.layers) {
      inOff.push(l.index === 0 ? 0 : postOff[l.index - 1]);
      postOff.push(actCursor);
      actCursor += batch * l.outDim;
      preOff.push(preCursor);
      preCursor += batch * l.outDim;
    }
    this.actOffsets = { inOff, preOff, postOff };

    const d = this.device;
    const S = GPUBufferUsage.STORAGE;
    this.acts = d.createBuffer({ size: actCursor * 4, usage: S });
    this.pre = d.createBuffer({ size: preCursor * 4, usage: S });
    this.deltas = d.createBuffer({ size: preCursor * 4, usage: S });
    // Sampled pixel indices in [0, batch), then one loss partial (bitcast f32)
    // per workgroup starting at `batch` — see the note above TRAIN_WGSL.
    const auxWords = batch + Math.ceil(batch / LOSS_WORKGROUP);
    this.aux = d.createBuffer({ size: auxWords * 4, usage: S | GPUBufferUsage.COPY_SRC });
    this.allocatedBatch = batch;

    this.trainBind0 = d.createBindGroup({
      layout: this.trainBGL0,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.moments } },
        { binding: 2, resource: { buffer: this.acts } },
        { binding: 3, resource: { buffer: this.pre } },
        { binding: 4, resource: { buffer: this.deltas } },
        { binding: 5, resource: { buffer: this.image } },
        { binding: 6, resource: { buffer: this.aux } },
        { binding: 7, resource: { buffer: this.encBuf } },
        { binding: 8, resource: { buffer: this.globalsBuf, size: 64 } },
      ],
    });

    this.writeLayerUniforms();
  }

  private writeLayerUniforms() {
    const layers = this.layout.layers;
    const buf = new ArrayBuffer(UNIFORM_STRIDE * layers.length);
    const dv = new DataView(buf);
    for (const l of layers) {
      const o = l.index * UNIFORM_STRIDE;
      const next = layers[l.index + 1];
      dv.setUint32(o + 0, l.inDim, true);
      dv.setUint32(o + 4, l.outDim, true);
      dv.setUint32(o + 8, l.wOff, true);
      dv.setUint32(o + 12, l.bOff, true);
      dv.setUint32(o + 16, this.actOffsets.inOff[l.index] ?? 0, true);
      dv.setUint32(o + 20, this.actOffsets.preOff[l.index] ?? 0, true);
      dv.setUint32(o + 24, this.actOffsets.postOff[l.index] ?? 0, true);
      dv.setUint32(o + 28, l.act, true);
      dv.setUint32(o + 32, next ? next.wOff : 0, true);
      dv.setUint32(o + 36, next ? next.outDim : 0, true);
      dv.setUint32(o + 40, next ? (this.actOffsets.preOff[next.index] ?? 0) : 0, true);
      dv.setFloat32(o + 48, l.omega, true);
    }
    this.device.queue.writeBuffer(this.layerBuf, 0, buf);
  }

  private writeGlobals(steps: number) {
    const buf = new ArrayBuffer(UNIFORM_STRIDE * steps);
    const dv = new DataView(buf);
    const h = this.hyper;
    for (let s = 0; s < steps; s++) {
      const o = s * UNIFORM_STRIDE;
      dv.setUint32(o + 0, h.batch, true);
      dv.setUint32(o + 4, this.imgW, true);
      dv.setUint32(o + 8, this.imgH, true);
      dv.setUint32(o + 12, Math.imul(this.stepCount + s + 1, 0x9e3779b1) >>> 0, true);
      dv.setFloat32(o + 16, h.lr, true);
      dv.setFloat32(o + 20, h.beta1, true);
      dv.setFloat32(o + 24, h.beta2, true);
      dv.setFloat32(o + 28, h.eps, true);
      dv.setFloat32(o + 32, this.stepCount + s + 1, true);
      dv.setUint32(o + 36, this.layout.paramCount, true);
      dv.setUint32(o + 40, ENC_CODE[this.layout.spec.encoding], true);
      dv.setUint32(o + 44, this.layout.spec.frequencies, true);
      dv.setUint32(o + 48, this.layout.inputDim, true);
      dv.setFloat32(o + 52, h.weightDecay, true);
    }
    this.device.queue.writeBuffer(this.globalsBuf, 0, buf);
  }

  // ---------------------------------------------------------------- config

  setOutputSize(requestedW: number, requestedH: number) {
    const stride = this.renderStride();

    // A wide model at a high field resolution can exceed the adapter's storage
    // binding limit; step the grid down until the ping-pong buffers fit.
    const limit = this.device.limits.maxStorageBufferBindingSize;
    let width = requestedW;
    let height = requestedH;
    while (width * height * stride * 4 > limit && width > 32) {
      width >>= 1;
      height >>= 1;
    }

    if (this.main && this.main.width === width && this.main.height === height) return;
    this.main?.destroy();
    this.main = new RenderTarget(this.device, width, height, stride);
    this.bindRenderTarget(this.main);
    this.canvas.width = width;
    this.canvas.height = height;

    const bg = new ArrayBuffer(16);
    const dv = new DataView(bg);
    dv.setUint32(0, width, true);
    dv.setUint32(4, height, true);
    this.device.queue.writeBuffer(this.blitUniform, 0, bg);
    this.blitBind = this.device.createBindGroup({
      layout: this.blitBGL,
      entries: [
        { binding: 0, resource: { buffer: this.main.outBuf } },
        { binding: 1, resource: { buffer: this.blitUniform } },
      ],
    });
  }

  private renderStride(): number {
    return Math.max(this.layout.inputDim, this.layout.spec.width, 4);
  }

  private bindRenderTarget(t: RenderTarget) {
    const d = this.device;
    t.bind0 = d.createBindGroup({
      layout: this.renderBGL0,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: this.encBuf } },
        { binding: 2, resource: { buffer: t.outBuf } },
        { binding: 3, resource: { buffer: t.uniform } },
      ],
    });
    t.bindAB = d.createBindGroup({
      layout: this.renderBGL1,
      entries: [
        { binding: 0, resource: { buffer: t.bufA } },
        { binding: 1, resource: { buffer: t.bufB } },
      ],
    });
    t.bindBA = d.createBindGroup({
      layout: this.renderBGL1,
      entries: [
        { binding: 0, resource: { buffer: t.bufB } },
        { binding: 1, resource: { buffer: t.bufA } },
      ],
    });
    t.writeUniform(this.layout, this.view);
  }

  setHyper(h: HyperParams) {
    const batchChanged = h.batch !== this.hyper.batch;
    this.hyper = { ...h };
    if (batchChanged) this.allocBatchBuffers(h.batch);
  }

  setView(view: ViewBox) {
    this.view = { ...view };
    this.main?.writeUniform(this.layout, this.view);
  }

  setImage(data: Float32Array, width: number, height: number) {
    this.imgW = width;
    this.imgH = height;
    writeF32(this.device.queue, this.image, data, width * height * 4);
  }

  getParams(): Float32Array {
    return this.cpuParams;
  }

  setParams(p: Float32Array) {
    this.cpuParams.set(p);
    this.uploadParams();
  }

  private uploadParams() {
    writeF32(this.device.queue, this.params, this.cpuParams);
  }

  /** Zero Adam's moments — used after a manual weight edit or a reinit. */
  resetOptimizer() {
    writeF32(this.device.queue, this.moments, new Float32Array(this.layout.paramCount * 2));
    this.stepCount = 0;
  }

  // ---------------------------------------------------------------- run

  tick(steps: number, render: boolean) {
    if (this.destroyed) return;
    const n = Math.max(0, Math.min(steps, MAX_STEPS_PER_FRAME));
    const enc = this.device.createCommandEncoder();

    if (n > 0) {
      this.writeGlobals(n);
      const pass = enc.beginComputePass();
      for (let s = 0; s < n; s++) this.encodeStep(pass, s * UNIFORM_STRIDE);
      pass.end();
      this.stepCount += n;
      // Every step overwrites the partials, so this reads the final step's loss.
      // The region starts right after the batchIdx slots within `aux`.
      this.lossPartials = Math.ceil(this.hyper.batch / LOSS_WORKGROUP);
      if (!this.lossPending) {
        enc.copyBufferToBuffer(
          this.aux,
          this.hyper.batch * 4,
          this.lossStage,
          0,
          this.lossPartials * 4,
        );
      }
    }

    if (render) this.encodeRender(enc, this.main, this.layout.layers.length, true);

    this.device.queue.submit([enc.finish()]);

    if (n > 0 && !this.lossPending) void this.readLoss();
  }

  private encodeStep(pass: GPUComputePassEncoder, gOff: number) {
    const layers = this.layout.layers;
    const batch = this.hyper.batch;
    const bind = (layerIndex: number) => {
      pass.setBindGroup(0, this.trainBind0, [gOff]);
      pass.setBindGroup(1, this.trainBind1, [layerIndex * UNIFORM_STRIDE]);
    };

    pass.setPipeline(this.pEncode);
    bind(0);
    pass.dispatchWorkgroups(Math.ceil(batch / 64));

    pass.setPipeline(this.pForward);
    for (const l of layers) {
      bind(l.index);
      pass.dispatchWorkgroups(Math.ceil(batch / 16), Math.ceil(l.outDim / 4));
    }

    const last = layers[layers.length - 1];
    pass.setPipeline(this.pOutputGrad);
    bind(last.index);
    pass.dispatchWorkgroups(Math.ceil(batch / 64));

    pass.setPipeline(this.pBackward);
    for (let i = layers.length - 2; i >= 0; i--) {
      bind(i);
      pass.dispatchWorkgroups(Math.ceil(batch / 16), Math.ceil(layers[i].outDim / 4));
    }

    pass.setPipeline(this.pWeightGradAdam);
    for (const l of layers) {
      bind(l.index);
      pass.dispatchWorkgroups(Math.ceil((l.outDim * (l.inDim + 1)) / 64));
    }
  }

  /** Forward-evaluate the field over `t`'s grid, stopping after `upto` layers. */
  private encodeRender(
    encoder: GPUCommandEncoder,
    t: RenderTarget,
    upto: number,
    blitToCanvas: boolean,
  ) {
    const layers = this.layout.layers;
    const pixels = t.width * t.height;
    const pass = encoder.beginComputePass();

    pass.setBindGroup(0, t.bind0);
    pass.setPipeline(this.pREncode);
    pass.setBindGroup(1, t.bindBA); // dst = bufA
    pass.setBindGroup(2, this.renderBind2, [0]);
    pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));

    pass.setPipeline(this.pRForward);
    for (let i = 0; i < upto; i++) {
      pass.setBindGroup(1, i % 2 === 0 ? t.bindAB : t.bindBA);
      pass.setBindGroup(2, this.renderBind2, [i * UNIFORM_STRIDE]);
      pass.dispatchWorkgroups(Math.ceil(pixels / 16), Math.ceil(layers[i].outDim / 4));
    }

    if (upto === layers.length) {
      pass.setPipeline(this.pRResolve);
      pass.setBindGroup(1, t.bindReadingLayer(upto - 1));
      pass.setBindGroup(2, this.renderBind2, [0]);
      pass.dispatchWorkgroups(Math.ceil(pixels / 64));
    }
    pass.end();

    if (blitToCanvas) {
      const rp = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: this.ctx.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      rp.setPipeline(this.pBlit);
      rp.setBindGroup(0, this.blitBind);
      rp.draw(3);
      rp.end();
    }
  }

  private async readLoss() {
    this.lossPending = true;
    try {
      const count = this.lossPartials;
      await this.lossStage.mapAsync(GPUMapMode.READ, 0, count * 4);
      if (this.destroyed) return;
      const partials = new Float32Array(this.lossStage.getMappedRange(0, count * 4).slice(0));
      this.lossStage.unmap();
      let sum = 0;
      for (let i = 0; i < count; i++) sum += partials[i];
      this.lastLoss = sum;
    } catch {
      // Device lost or buffer destroyed mid-flight; keep the previous value.
    } finally {
      this.lossPending = false;
    }
  }

  loss() {
    return this.lastLoss;
  }

  async refreshParams(): Promise<Float32Array> {
    if (this.destroyed || this.paramReadPending) return this.cpuParams;
    this.paramReadPending = true;
    try {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.params, 0, this.paramStage, 0, this.layout.paramCount * 4);
      this.device.queue.submit([enc.finish()]);
      await this.paramStage.mapAsync(GPUMapMode.READ);
      if (this.destroyed) return this.cpuParams;
      this.cpuParams.set(new Float32Array(this.paramStage.getMappedRange()));
      this.paramStage.unmap();
    } catch {
      /* keep the stale mirror */
    } finally {
      this.paramReadPending = false;
    }
    return this.cpuParams;
  }

  async neuronPreview(layerIndex: number, size: number): Promise<NeuronPreview | null> {
    if (this.destroyed || this.previewPending) return null;
    const layer = this.layout.layers[layerIndex];
    if (!layer) return null;

    const stride = this.renderStride();
    if (!this.preview || this.preview.width !== size) {
      this.preview?.destroy();
      this.previewStage?.destroy();
      this.preview = new RenderTarget(this.device, size, size, stride);
      this.bindRenderTarget(this.preview);
      this.previewStage = this.device.createBuffer({
        size: size * size * stride * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    // Neuron thumbnails always show the canonical [-1, 1] domain.
    this.preview.writeUniform(this.layout, { zoom: 1, cx: 0, cy: 0 });

    this.previewPending = true;
    try {
      const enc = this.device.createCommandEncoder();
      this.encodeRender(enc, this.preview, layerIndex + 1, false);
      enc.copyBufferToBuffer(
        this.preview.bufferAfterLayer(layerIndex),
        0,
        this.previewStage,
        0,
        size * size * stride * 4,
      );
      this.device.queue.submit([enc.finish()]);
      await this.previewStage.mapAsync(GPUMapMode.READ);
      if (this.destroyed) return null;
      const data = new Float32Array(this.previewStage.getMappedRange().slice(0));
      this.previewStage.unmap();
      return { width: size, height: size, data, stride, count: layer.outDim };
    } catch {
      return null;
    } finally {
      this.previewPending = false;
    }
  }

  destroy() {
    this.destroyed = true;
    for (const b of [
      this.params,
      this.moments,
      this.acts,
      this.pre,
      this.deltas,
      this.image,
      this.aux,
      this.encBuf,
      this.globalsBuf,
      this.layerBuf,
      this.blitUniform,
    ]) {
      b?.destroy();
    }
    // Mapped-at-destroy staging buffers throw; ignore.
    for (const b of [this.lossStage, this.paramStage, this.previewStage]) {
      try {
        b?.destroy();
      } catch {
        /* already mapped */
      }
    }
    this.main?.destroy();
    this.preview?.destroy();
  }
}
