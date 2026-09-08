/**
 * Definition of the neural field g(x, y) -> rgb, and the flat parameter layout
 * shared by the CPU and WebGPU trainers.
 *
 * The network is a plain MLP applied per-pixel:
 *
 *     features = encode(x, y)            (fixed, not trained)
 *     h_0      = act(W_0 · features + b_0)
 *     ...
 *     rgb      = sigmoid(W_L · h_{L-1} + b_L)
 *
 * All weights and biases live in one Float32Array so they can be handed to the
 * GPU as a single storage buffer, and read back for visualisation/editing.
 */

export type Activation = 'relu' | 'tanh' | 'sine' | 'gelu';
export type Encoding = 'none' | 'positional' | 'gaussian';

export const ACTIVATIONS: Activation[] = ['relu', 'tanh', 'sine', 'gelu'];
export const ENCODINGS: Encoding[] = ['none', 'positional', 'gaussian'];

/** Activation codes shared with WGSL (see `activate` / `dActivate`). */
export const ACT_CODE: Record<Activation, number> = { relu: 0, tanh: 1, sine: 2, gelu: 3 };
export const SIGMOID_CODE = 4;

export const ENC_CODE: Record<Encoding, number> = { none: 0, positional: 1, gaussian: 2 };

export const OUTPUT_CHANNELS = 3;

export interface ModelSpec {
  /** Number of hidden layers. Total weight matrices is this + 1. */
  hiddenLayers: number;
  /** Neurons per hidden layer. */
  width: number;
  activation: Activation;
  encoding: Encoding;
  /** Number of frequency bands (positional) or random directions (gaussian). */
  frequencies: number;
  /** Bandwidth: octave multiplier for positional, sigma for gaussian. */
  encodingScale: number;
  /** Frequency multiplier for `sine` activations (SIREN's omega_0). */
  omega0: number;
  /** Seed for weight init + the gaussian encoding matrix. */
  seed: number;
}

export const DEFAULT_SPEC: ModelSpec = {
  hiddenLayers: 3,
  width: 64,
  activation: 'relu',
  encoding: 'gaussian',
  frequencies: 32,
  encodingScale: 4,
  omega0: 30,
  seed: 1,
};

export interface LayerInfo {
  index: number;
  inDim: number;
  outDim: number;
  /** Offset of this layer's weight matrix (row-major, [outDim][inDim]). */
  wOff: number;
  /** Offset of this layer's bias vector (length outDim). */
  bOff: number;
  act: number;
  omega: number;
  isOutput: boolean;
}

export interface ModelLayout {
  spec: ModelSpec;
  /** Dimensionality of encode(x, y). */
  inputDim: number;
  layers: LayerInfo[];
  paramCount: number;
  /**
   * Fixed encoding constants uploaded alongside the weights.
   *  - positional: one angular frequency per band.
   *  - gaussian:   one (bx, by) angular-frequency vector per row.
   */
  encData: Float32Array;
  /** Human-readable name for each input feature, for the architecture view. */
  featureLabels: string[];
}

export function featureDim(spec: ModelSpec): number {
  switch (spec.encoding) {
    case 'none':
      return 2;
    case 'positional':
      return 2 + 4 * spec.frequencies;
    case 'gaussian':
      return 2 * spec.frequencies;
  }
}

/** Deterministic PRNG so a given seed always rebuilds the same model. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianPair(rand: () => number): [number, number] {
  // Box-Muller. `rand()` can return 0, which would blow up the log.
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  const r = Math.sqrt(-2 * Math.log(u));
  return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)];
}

export function buildLayout(spec: ModelSpec): ModelLayout {
  const inputDim = featureDim(spec);
  const dims: number[] = [inputDim];
  for (let i = 0; i < spec.hiddenLayers; i++) dims.push(spec.width);
  dims.push(OUTPUT_CHANNELS);

  const layers: LayerInfo[] = [];
  let off = 0;
  for (let i = 0; i + 1 < dims.length; i++) {
    const inDim = dims[i];
    const outDim = dims[i + 1];
    const isOutput = i === dims.length - 2;
    const wOff = off;
    off += inDim * outDim;
    const bOff = off;
    off += outDim;
    layers.push({
      index: i,
      inDim,
      outDim,
      wOff,
      bOff,
      act: isOutput ? SIGMOID_CODE : ACT_CODE[spec.activation],
      // omega_0 only scales the *first* layer in the SIREN formulation.
      omega: spec.activation === 'sine' && !isOutput ? (i === 0 ? spec.omega0 : 1) : 1,
      isOutput,
    });
  }

  const rand = mulberry32(spec.seed ^ 0x9e3779b9);
  const { encData, featureLabels } = buildEncoding(spec, rand);

  return { spec, inputDim, layers, paramCount: off, encData, featureLabels };
}

function buildEncoding(spec: ModelSpec, rand: () => number) {
  const labels: string[] = [];
  if (spec.encoding === 'none') {
    labels.push('x', 'y');
    return { encData: new Float32Array(0), featureLabels: labels };
  }

  if (spec.encoding === 'positional') {
    // sin/cos(2^k · scale · π · p) for each axis, plus the raw coordinates.
    const f = new Float32Array(spec.frequencies);
    labels.push('x', 'y');
    for (let k = 0; k < spec.frequencies; k++) {
      f[k] = Math.PI * spec.encodingScale * Math.pow(2, k / 2);
      const w = (f[k] / Math.PI).toFixed(1);
      labels.push(`sin ${w}πx`, `cos ${w}πx`, `sin ${w}πy`, `cos ${w}πy`);
    }
    return { encData: f, featureLabels: labels };
  }

  // Gaussian random Fourier features (Tancik et al. 2020): rows of B ~ N(0, σ²),
  // features = [sin(2π B p), cos(2π B p)]. The 2π is baked into the stored values.
  const b = new Float32Array(spec.frequencies * 2);
  for (let k = 0; k < spec.frequencies; k++) {
    const [g0, g1] = gaussianPair(rand);
    b[k * 2] = 2 * Math.PI * spec.encodingScale * g0;
    b[k * 2 + 1] = 2 * Math.PI * spec.encodingScale * g1;
    const mag = Math.hypot(g0, g1) * spec.encodingScale;
    labels.push(`sin b${k} (|b|=${mag.toFixed(1)})`, `cos b${k}`);
  }
  return { encData: b, featureLabels: labels };
}

/** Fresh weights for `layout`, using the initialisation appropriate to the activation. */
export function initParams(layout: ModelLayout): Float32Array {
  const p = new Float32Array(layout.paramCount);
  const rand = mulberry32(layout.spec.seed);
  const act = layout.spec.activation;

  for (const l of layout.layers) {
    if (act === 'sine' && !l.isOutput) {
      // SIREN init: first layer U(-1/n, 1/n), later layers U(±sqrt(6/n)/omega).
      const bound = l.index === 0 ? 1 / l.inDim : Math.sqrt(6 / l.inDim) / l.omega;
      for (let i = 0; i < l.inDim * l.outDim; i++) p[l.wOff + i] = (rand() * 2 - 1) * bound;
    } else {
      // He for relu/gelu, Xavier for tanh and the sigmoid output.
      const gain = act === 'relu' || act === 'gelu' ? 2 : 1;
      const std = Math.sqrt(gain / l.inDim);
      for (let i = 0; i < l.inDim * l.outDim; i += 2) {
        const [g0, g1] = gaussianPair(rand);
        p[l.wOff + i] = g0 * std;
        if (i + 1 < l.inDim * l.outDim) p[l.wOff + i + 1] = g1 * std;
      }
    }
    // Biases start at zero; the output bias at 0 puts sigmoid at mid-grey.
    for (let j = 0; j < l.outDim; j++) p[l.bOff + j] = 0;
  }
  return p;
}

/** Evaluate encode(x, y) into `out` at `base`. Mirrors `writeFeatures` in WGSL. */
export function encodeInto(
  layout: ModelLayout,
  x: number,
  y: number,
  out: Float32Array,
  base: number,
): void {
  const { spec, encData } = layout;
  if (spec.encoding === 'gaussian') {
    for (let k = 0; k < spec.frequencies; k++) {
      const d = encData[k * 2] * x + encData[k * 2 + 1] * y;
      out[base + k * 2] = Math.sin(d);
      out[base + k * 2 + 1] = Math.cos(d);
    }
    return;
  }
  out[base] = x;
  out[base + 1] = y;
  if (spec.encoding === 'positional') {
    for (let k = 0; k < spec.frequencies; k++) {
      const f = encData[k];
      const o = base + 2 + k * 4;
      out[o] = Math.sin(f * x);
      out[o + 1] = Math.cos(f * x);
      out[o + 2] = Math.sin(f * y);
      out[o + 3] = Math.cos(f * y);
    }
  }
}

export function describeSpec(layout: ModelLayout): string {
  const dims = [layout.inputDim, ...layout.layers.map((l) => l.outDim)];
  return dims.join(' → ');
}
