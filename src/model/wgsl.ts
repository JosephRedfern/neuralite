/**
 * WGSL sources for the trainer.
 *
 * The MLP is small (a few thousand to ~150k parameters), so rather than a
 * generic matmul we use one thread per output element and loop over the
 * contracted dimension. Per-layer constants arrive in a uniform buffer bound
 * with a dynamic offset, so the whole training step is one pipeline set and a
 * handful of dispatches — no shader recompilation when the model changes.
 */

/** Shared prelude: bindings, activations, hashing. */
const COMMON = /* wgsl */ `
struct Globals {
  batch      : u32,
  imgW       : u32,
  imgH       : u32,
  seed       : u32,
  lr         : f32,
  beta1      : f32,
  beta2      : f32,
  eps        : f32,
  t          : f32,
  paramCount : u32,
  encMode    : u32,
  encCount   : u32,
  inputDim   : u32,
  weightDecay: f32,
  _pad0      : f32,
  _pad1      : f32,
};

struct LayerU {
  inDim       : u32,
  outDim      : u32,
  wOff        : u32,
  bOff        : u32,
  inOff       : u32,
  preOff      : u32,
  postOff     : u32,
  act         : u32,
  nextWOff    : u32,
  nextOutDim  : u32,
  nextPreOff  : u32,
  _pad0       : u32,
  omega       : f32,
  _pad1       : f32,
  _pad2       : f32,
  _pad3       : f32,
};

fn activate(x: f32, a: u32, w: f32) -> f32 {
  if (a == 0u) { return max(x, 0.0); }
  if (a == 1u) { return tanh(x); }
  if (a == 2u) { return sin(w * x); }
  if (a == 3u) {
    let t = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
  }
  return 1.0 / (1.0 + exp(-x));
}

/** Derivative of activate w.r.t. its pre-activation, given pre x and post y. */
fn dActivate(x: f32, y: f32, a: u32, w: f32) -> f32 {
  if (a == 0u) { return select(0.0, 1.0, x > 0.0); }
  if (a == 1u) { return 1.0 - y * y; }
  if (a == 2u) { return w * cos(w * x); }
  if (a == 3u) {
    let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    let t = tanh(inner);
    let dInner = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dInner;
  }
  return y * (1.0 - y);
}

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
`;

/**
 * WebGPU only guarantees 8 storage buffers per compute stage (that used to be
 * 11 here: params, grads, mom1, mom2, acts, pre, deltas, image, batchIdx, enc,
 * loss — one bind group over the limit fails pipeline creation outright, which
 * silently invalidates every command buffer built from it). Three buffers are
 * folded together to fit the guaranteed minimum on any conformant device:
 *  - mom1/mom2   -> one \`moments\` buffer, mom2 living at +paramCount.
 *  - batchIdx and the per-workgroup loss partials -> one \`aux\` u32 buffer,
 *    the loss region living at +batch (loss values are bitcast to u32).
 *  - grads is gone entirely: weightGrad and the Adam step are fused into one
 *    kernel, since each parameter's update depends on nothing but its own
 *    gradient.
 */
export const TRAIN_WGSL = /* wgsl */ `
${COMMON}

@group(0) @binding(0) var<storage, read_write> params  : array<f32>;
@group(0) @binding(1) var<storage, read_write> moments : array<f32>;
@group(0) @binding(2) var<storage, read_write> acts    : array<f32>;
@group(0) @binding(3) var<storage, read_write> pre     : array<f32>;
@group(0) @binding(4) var<storage, read_write> deltas  : array<f32>;
@group(0) @binding(5) var<storage, read>       image   : array<f32>;
@group(0) @binding(6) var<storage, read_write> aux     : array<u32>;
@group(0) @binding(7) var<storage, read>       enc     : array<f32>;
@group(0) @binding(8) var<uniform>             G       : Globals;

@group(1) @binding(0) var<uniform>             L       : LayerU;

fn writeFeatures(x: f32, y: f32, base: u32) {
  if (G.encMode == 2u) {
    for (var k = 0u; k < G.encCount; k = k + 1u) {
      let d = enc[k * 2u] * x + enc[k * 2u + 1u] * y;
      acts[base + k * 2u]      = sin(d);
      acts[base + k * 2u + 1u] = cos(d);
    }
    return;
  }
  acts[base]      = x;
  acts[base + 1u] = y;
  if (G.encMode == 1u) {
    for (var k = 0u; k < G.encCount; k = k + 1u) {
      let f = enc[k];
      let o = base + 2u + k * 4u;
      acts[o]      = sin(f * x);
      acts[o + 1u] = cos(f * x);
      acts[o + 2u] = sin(f * y);
      acts[o + 3u] = cos(f * y);
    }
  }
}

/** Pick a random pixel per batch slot and write its encoded coordinates. */
@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= G.batch) { return; }

  let n = G.imgW * G.imgH;
  let idx = pcg(G.seed + b * 2654435761u) % n;
  aux[b] = idx;

  let px = f32(idx % G.imgW);
  let py = f32(idx / G.imgW);
  let x = ((px + 0.5) / f32(G.imgW)) * 2.0 - 1.0;
  let y = ((py + 0.5) / f32(G.imgH)) * 2.0 - 1.0;
  writeFeatures(x, y, b * G.inputDim);
}

@compute @workgroup_size(16, 4)
fn forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  let j = gid.y;
  if (b >= G.batch || j >= L.outDim) { return; }

  let inBase = L.inOff + b * L.inDim;
  let wBase = L.wOff + j * L.inDim;
  var s = params[L.bOff + j];
  for (var i = 0u; i < L.inDim; i = i + 1u) {
    s = s + acts[inBase + i] * params[wBase + i];
  }
  pre[L.preOff + b * L.outDim + j] = s;
  acts[L.postOff + b * L.outDim + j] = activate(s, L.act, L.omega);
}

var<workgroup> lossShared : array<f32, 64>;

/** dL/d(pre) for the output layer, plus one loss partial per workgroup. */
@compute @workgroup_size(64)
fn outputGrad(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let b = gid.x;
  let norm = 1.0 / f32(G.batch * L.outDim);
  var sqErr = 0.0;

  // No early return: every invocation must reach the barriers below.
  if (b < G.batch) {
    let idx = aux[b];
    for (var c = 0u; c < L.outDim; c = c + 1u) {
      let o = b * L.outDim + c;
      let y = acts[L.postOff + o];
      let diff = y - image[idx * 4u + c];
      sqErr = sqErr + diff * diff;
      deltas[L.preOff + o] =
        2.0 * diff * norm * dActivate(pre[L.preOff + o], y, L.act, L.omega);
    }
  }

  // Tree-reduce in f32 rather than atomically accumulating fixed point: at
  // large batches a single sample's share of the mean underflows any u32 scale
  // that is also overflow-safe.
  lossShared[lid.x] = sqErr * norm;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      lossShared[lid.x] = lossShared[lid.x] + lossShared[lid.x + s];
    }
    workgroupBarrier();
  }
  // Loss partials live past the batchIdx region of \`aux\`, one f32 (bitcast)
  // per workgroup — this frame's final step overwrites them, so a readback
  // always sees the freshest values.
  if (lid.x == 0u) {
    aux[G.batch + wid.x] = bitcast<u32>(lossShared[0]);
  }
}

/** Backprop through layer L using the *next* layer's weights and deltas. */
@compute @workgroup_size(16, 4)
fn backwardDelta(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  let i = gid.y;
  if (b >= G.batch || i >= L.outDim) { return; }

  let dBase = L.nextPreOff + b * L.nextOutDim;
  var s = 0.0;
  for (var j = 0u; j < L.nextOutDim; j = j + 1u) {
    s = s + deltas[dBase + j] * params[L.nextWOff + j * L.outDim + i];
  }
  let o = L.preOff + b * L.outDim + i;
  s = s * dActivate(pre[o], acts[L.postOff + b * L.outDim + i], L.act, L.omega);
  deltas[o] = s;
}

/**
 * One thread per parameter of this layer: reduce its gradient over the batch,
 * then apply the Adam update immediately. Fused with what used to be a
 * separate \`adam\` pass over all layers — Adam is elementwise, so updating a
 * layer's parameters the moment its gradient is known is equivalent to a
 * later global pass, and it saves a whole extra buffer (see the note above
 * TRAIN_WGSL) plus a dispatch. Column index == inDim addresses the bias.
 */
@compute @workgroup_size(64)
fn weightGradAdam(@builtin(global_invocation_id) gid: vec3<u32>) {
  let stride = L.inDim + 1u;
  let t = gid.x;
  if (t >= L.outDim * stride) { return; }

  let j = t / stride;
  let i = t % stride;
  var g = 0.0;
  var paramIdx: u32;

  if (i == L.inDim) {
    for (var b = 0u; b < G.batch; b = b + 1u) {
      g = g + deltas[L.preOff + b * L.outDim + j];
    }
    paramIdx = L.bOff + j;
  } else {
    for (var b = 0u; b < G.batch; b = b + 1u) {
      g = g + deltas[L.preOff + b * L.outDim + j] * acts[L.inOff + b * L.inDim + i];
    }
    paramIdx = L.wOff + j * L.inDim + i;
  }

  g = g + G.weightDecay * params[paramIdx];
  let m = G.beta1 * moments[paramIdx] + (1.0 - G.beta1) * g;
  let v = G.beta2 * moments[G.paramCount + paramIdx] + (1.0 - G.beta2) * g * g;
  moments[paramIdx] = m;
  moments[G.paramCount + paramIdx] = v;

  let mHat = m / (1.0 - pow(G.beta1, G.t));
  let vHat = v / (1.0 - pow(G.beta2, G.t));
  params[paramIdx] = params[paramIdx] - G.lr * mHat / (sqrt(vHat) + G.eps);
}
`;

export const RENDER_WGSL = /* wgsl */ `
${COMMON}

struct RenderG {
  w        : u32,
  h        : u32,
  inputDim : u32,
  encMode  : u32,
  encCount : u32,
  stride   : u32,
  _pad0    : u32,
  _pad1    : u32,
  zoom     : f32,
  cx       : f32,
  cy       : f32,
  _pad2    : f32,
};

@group(0) @binding(0) var<storage, read> params : array<f32>;
@group(0) @binding(1) var<storage, read> enc    : array<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf : array<f32>;
@group(0) @binding(3) var<uniform> R : RenderG;

@group(1) @binding(0) var<storage, read>       src : array<f32>;
@group(1) @binding(1) var<storage, read_write> dst : array<f32>;

@group(2) @binding(0) var<uniform> L : LayerU;

fn writeFeaturesR(x: f32, y: f32, base: u32) {
  if (R.encMode == 2u) {
    for (var k = 0u; k < R.encCount; k = k + 1u) {
      let d = enc[k * 2u] * x + enc[k * 2u + 1u] * y;
      dst[base + k * 2u]      = sin(d);
      dst[base + k * 2u + 1u] = cos(d);
    }
    return;
  }
  dst[base]      = x;
  dst[base + 1u] = y;
  if (R.encMode == 1u) {
    for (var k = 0u; k < R.encCount; k = k + 1u) {
      let f = enc[k];
      let o = base + 2u + k * 4u;
      dst[o]      = sin(f * x);
      dst[o + 1u] = cos(f * x);
      dst[o + 2u] = sin(f * y);
      dst[o + 3u] = cos(f * y);
    }
  }
}

@compute @workgroup_size(8, 8)
fn rEncode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= R.w || gid.y >= R.h) { return; }
  let p = gid.y * R.w + gid.x;
  let x = R.cx + (((f32(gid.x) + 0.5) / f32(R.w)) * 2.0 - 1.0) * R.zoom;
  let y = R.cy + (((f32(gid.y) + 0.5) / f32(R.h)) * 2.0 - 1.0) * R.zoom;
  writeFeaturesR(x, y, p * R.stride);
}

@compute @workgroup_size(16, 4)
fn rForward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  let j = gid.y;
  let total = R.w * R.h;
  if (p >= total || j >= L.outDim) { return; }

  let inBase = p * R.stride;
  let wBase = L.wOff + j * L.inDim;
  var s = params[L.bOff + j];
  for (var i = 0u; i < L.inDim; i = i + 1u) {
    s = s + src[inBase + i] * params[wBase + i];
  }
  dst[inBase + j] = activate(s, L.act, L.omega);
}

/** Copy the final rgb activations into the packed output buffer. */
@compute @workgroup_size(64)
fn rResolve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= R.w * R.h) { return; }
  outBuf[p * 4u + 0u] = src[p * R.stride + 0u];
  outBuf[p * 4u + 1u] = src[p * R.stride + 1u];
  outBuf[p * 4u + 2u] = src[p * R.stride + 2u];
  outBuf[p * 4u + 3u] = 1.0;
}
`;

/** Fullscreen blit of the packed render buffer onto the canvas. */
export const BLIT_WGSL = /* wgsl */ `
struct BlitG { w: u32, h: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<storage, read> outBuf : array<f32>;
@group(0) @binding(1) var<uniform> B : BlitG;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) c: vec4<f32>) -> @location(0) vec4<f32> {
  let x = min(u32(c.x), B.w - 1u);
  let y = min(u32(c.y), B.h - 1u);
  let o = (y * B.w + x) * 4u;
  return vec4<f32>(outBuf[o], outBuf[o + 1u], outBuf[o + 2u], 1.0);
}
`;
