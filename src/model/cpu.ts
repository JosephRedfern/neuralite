import {
  buildLayout,
  encodeInto,
  initParams,
  SIGMOID_CODE,
  type ModelLayout,
  type ModelSpec,
} from './spec';
import type { HyperParams, NeuronPreview, Trainer, ViewBox } from './trainer';

/**
 * Reference implementation of the same training step as `gpu.ts`, in plain
 * TypedArrays. Used when WebGPU is unavailable, and useful for checking the
 * shaders: identical seeds and hyperparameters should track each other closely.
 */

function activate(x: number, a: number, w: number): number {
  switch (a) {
    case 0:
      return x > 0 ? x : 0;
    case 1:
      return Math.tanh(x);
    case 2:
      return Math.sin(w * x);
    case 3: {
      const t = Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x));
      return 0.5 * x * (1 + t);
    }
    default:
      return 1 / (1 + Math.exp(-x));
  }
}

function dActivate(x: number, y: number, a: number, w: number): number {
  switch (a) {
    case 0:
      return x > 0 ? 1 : 0;
    case 1:
      return 1 - y * y;
    case 2:
      return w * Math.cos(w * x);
    case 3: {
      const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
      const t = Math.tanh(inner);
      const dInner = 0.7978845608 * (1 + 3 * 0.044715 * x * x);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner;
    }
    default:
      return y * (1 - y);
  }
}

export class CpuTrainer implements Trainer {
  readonly backend = 'cpu' as const;
  readonly canvas: HTMLCanvasElement;
  readonly layout: ModelLayout;

  private ctx: CanvasRenderingContext2D;
  private imageData!: ImageData;
  private outW = 0;
  private outH = 0;

  private params: Float32Array;
  private grads: Float32Array;
  private mom1: Float32Array;
  private mom2: Float32Array;

  private acts: Float32Array = new Float32Array(0);
  private pre: Float32Array = new Float32Array(0);
  private deltas: Float32Array = new Float32Array(0);
  private batchIdx: Int32Array = new Int32Array(0);
  private inOff: number[] = [];
  private preOff: number[] = [];
  private postOff: number[] = [];
  private allocatedBatch = 0;

  private image = new Float32Array(4);
  private imgW = 1;
  private imgH = 1;

  private hyper: HyperParams;
  private view: ViewBox = { zoom: 1, cx: 0, cy: 0 };
  private lastLoss = 0;
  private rngState = 12345;

  stepCount = 0;

  constructor(
    canvas: HTMLCanvasElement,
    spec: ModelSpec,
    hyper: HyperParams,
    outW: number,
    outH: number,
  ) {
    this.canvas = canvas;
    this.layout = buildLayout(spec);
    this.hyper = { ...hyper };
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('could not acquire a 2d canvas context');
    this.ctx = ctx;

    const n = this.layout.paramCount;
    this.params = initParams(this.layout);
    this.grads = new Float32Array(n);
    this.mom1 = new Float32Array(n);
    this.mom2 = new Float32Array(n);

    this.alloc(hyper.batch);
    this.setOutputSize(outW, outH);
  }

  private alloc(batch: number) {
    if (batch === this.allocatedBatch) return;
    this.inOff = [];
    this.preOff = [];
    this.postOff = [];
    let actCursor = batch * this.layout.inputDim;
    let preCursor = 0;
    for (const l of this.layout.layers) {
      this.inOff.push(l.index === 0 ? 0 : this.postOff[l.index - 1]);
      this.postOff.push(actCursor);
      actCursor += batch * l.outDim;
      this.preOff.push(preCursor);
      preCursor += batch * l.outDim;
    }
    this.acts = new Float32Array(actCursor);
    this.pre = new Float32Array(preCursor);
    this.deltas = new Float32Array(preCursor);
    this.batchIdx = new Int32Array(batch);
    this.allocatedBatch = batch;
  }

  private nextRand(): number {
    // xorshift32
    let x = this.rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x >>> 0;
    return this.rngState;
  }

  setHyper(h: HyperParams) {
    const batchChanged = h.batch !== this.hyper.batch;
    this.hyper = { ...h };
    if (batchChanged) this.alloc(h.batch);
  }

  setView(view: ViewBox) {
    this.view = { ...view };
  }

  setImage(data: Float32Array, width: number, height: number) {
    if (this.image.length < width * height * 4) this.image = new Float32Array(width * height * 4);
    this.image.set(data.subarray(0, width * height * 4));
    this.imgW = width;
    this.imgH = height;
  }

  setOutputSize(width: number, height: number) {
    if (this.outW === width && this.outH === height) return;
    this.outW = width;
    this.outH = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.imageData = this.ctx.createImageData(width, height);
  }

  getParams() {
    return this.params;
  }

  setParams(p: Float32Array) {
    this.params.set(p);
  }

  async refreshParams() {
    return this.params;
  }

  resetOptimizer() {
    this.mom1.fill(0);
    this.mom2.fill(0);
    this.stepCount = 0;
  }

  loss() {
    return this.lastLoss;
  }

  tick(steps: number, render: boolean) {
    let total = 0;
    for (let s = 0; s < steps; s++) total += this.step();
    if (steps > 0) this.lastLoss = total / steps;
    if (render) this.render();
  }

  private step(): number {
    const { params, acts, pre, deltas, grads } = this;
    const batch = this.hyper.batch;
    const layers = this.layout.layers;
    const inputDim = this.layout.inputDim;
    const nPixels = this.imgW * this.imgH;

    // Sample a batch of pixels and encode their coordinates.
    for (let b = 0; b < batch; b++) {
      const idx = this.nextRand() % nPixels;
      this.batchIdx[b] = idx;
      const px = idx % this.imgW;
      const py = (idx / this.imgW) | 0;
      const x = ((px + 0.5) / this.imgW) * 2 - 1;
      const y = ((py + 0.5) / this.imgH) * 2 - 1;
      encodeInto(this.layout, x, y, acts, b * inputDim);
    }

    // Forward.
    for (const l of layers) {
      const { inDim, outDim, wOff, bOff } = l;
      const iOff = this.inOff[l.index];
      const pOff = this.preOff[l.index];
      const oOff = this.postOff[l.index];
      for (let b = 0; b < batch; b++) {
        const inBase = iOff + b * inDim;
        for (let j = 0; j < outDim; j++) {
          const wBase = wOff + j * inDim;
          let s = params[bOff + j];
          for (let i = 0; i < inDim; i++) s += acts[inBase + i] * params[wBase + i];
          pre[pOff + b * outDim + j] = s;
          acts[oOff + b * outDim + j] = activate(s, l.act, l.omega);
        }
      }
    }

    // Output layer gradient + loss.
    const last = layers[layers.length - 1];
    const lastPre = this.preOff[last.index];
    const lastPost = this.postOff[last.index];
    const norm = 1 / (batch * last.outDim);
    let loss = 0;
    for (let b = 0; b < batch; b++) {
      const idx = this.batchIdx[b];
      for (let c = 0; c < last.outDim; c++) {
        const o = b * last.outDim + c;
        const y = acts[lastPost + o];
        const diff = y - this.image[idx * 4 + c];
        loss += diff * diff;
        deltas[lastPre + o] =
          2 * diff * norm * dActivate(pre[lastPre + o], y, SIGMOID_CODE, 1);
      }
    }
    loss *= norm;

    // Backward through the hidden layers.
    for (let li = layers.length - 2; li >= 0; li--) {
      const l = layers[li];
      const next = layers[li + 1];
      const pOff = this.preOff[li];
      const oOff = this.postOff[li];
      const nOff = this.preOff[li + 1];
      for (let b = 0; b < batch; b++) {
        const dBase = nOff + b * next.outDim;
        for (let i = 0; i < l.outDim; i++) {
          let s = 0;
          for (let j = 0; j < next.outDim; j++) {
            s += deltas[dBase + j] * params[next.wOff + j * l.outDim + i];
          }
          const o = pOff + b * l.outDim + i;
          deltas[o] = s * dActivate(pre[o], acts[oOff + b * l.outDim + i], l.act, l.omega);
        }
      }
    }

    // Parameter gradients.
    grads.fill(0);
    for (const l of layers) {
      const { inDim, outDim, wOff, bOff } = l;
      const iOff = this.inOff[l.index];
      const pOff = this.preOff[l.index];
      for (let b = 0; b < batch; b++) {
        const inBase = iOff + b * inDim;
        const dBase = pOff + b * outDim;
        for (let j = 0; j < outDim; j++) {
          const d = deltas[dBase + j];
          if (d === 0) continue;
          const wBase = wOff + j * inDim;
          for (let i = 0; i < inDim; i++) grads[wBase + i] += acts[inBase + i] * d;
          grads[bOff + j] += d;
        }
      }
    }

    // Adam.
    this.stepCount++;
    const { lr, beta1, beta2, eps, weightDecay } = this.hyper;
    const bc1 = 1 - Math.pow(beta1, this.stepCount);
    const bc2 = 1 - Math.pow(beta2, this.stepCount);
    for (let p = 0; p < params.length; p++) {
      const g = grads[p] + weightDecay * params[p];
      const m = beta1 * this.mom1[p] + (1 - beta1) * g;
      const v = beta2 * this.mom2[p] + (1 - beta2) * g * g;
      this.mom1[p] = m;
      this.mom2[p] = v;
      params[p] -= (lr * (m / bc1)) / (Math.sqrt(v / bc2) + eps);
    }

    return loss;
  }

  /** Forward pass over a grid, returning the activations of layer `upto - 1`. */
  private forwardGrid(w: number, h: number, upto: number, view: ViewBox) {
    const stride = Math.max(this.layout.inputDim, this.layout.spec.width, 4);
    const n = w * h;
    let src = new Float32Array(n * stride);
    let dst = new Float32Array(n * stride);

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const x = view.cx + (((px + 0.5) / w) * 2 - 1) * view.zoom;
        const y = view.cy + (((py + 0.5) / h) * 2 - 1) * view.zoom;
        encodeInto(this.layout, x, y, src, (py * w + px) * stride);
      }
    }

    for (let li = 0; li < upto; li++) {
      const l = this.layout.layers[li];
      for (let p = 0; p < n; p++) {
        const base = p * stride;
        for (let j = 0; j < l.outDim; j++) {
          const wBase = l.wOff + j * l.inDim;
          let s = this.params[l.bOff + j];
          for (let i = 0; i < l.inDim; i++) s += src[base + i] * this.params[wBase + i];
          dst[base + j] = activate(s, l.act, l.omega);
        }
      }
      const tmp = src;
      src = dst;
      dst = tmp;
    }
    return { data: src, stride };
  }

  private render() {
    const { data, stride } = this.forwardGrid(
      this.outW,
      this.outH,
      this.layout.layers.length,
      this.view,
    );
    const px = this.imageData.data;
    for (let p = 0; p < this.outW * this.outH; p++) {
      px[p * 4] = Math.max(0, Math.min(255, data[p * stride] * 255));
      px[p * 4 + 1] = Math.max(0, Math.min(255, data[p * stride + 1] * 255));
      px[p * 4 + 2] = Math.max(0, Math.min(255, data[p * stride + 2] * 255));
      px[p * 4 + 3] = 255;
    }
    this.ctx.putImageData(this.imageData, 0, 0);
  }

  async neuronPreview(layerIndex: number, size: number): Promise<NeuronPreview | null> {
    const layer = this.layout.layers[layerIndex];
    if (!layer) return null;
    const { data, stride } = this.forwardGrid(size, size, layerIndex + 1, {
      zoom: 1,
      cx: 0,
      cy: 0,
    });
    return { width: size, height: size, data, stride, count: layer.outDim };
  }

  destroy() {
    /* nothing device-side to release */
  }
}
