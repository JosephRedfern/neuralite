/**
 * Numerical checks for the reference (CPU) trainer, which the WGSL kernels
 * mirror line for line.
 *
 *   npx tsx scripts/check.ts
 *
 * 1. Finite-difference gradient check across every activation/encoding combo.
 * 2. An end-to-end fit on a synthetic target, asserting the loss actually falls.
 */

import { CpuTrainer } from '../src/model/cpu';
import { DEFAULT_HYPER } from '../src/model/trainer';
import { ACTIVATIONS, ENCODINGS, type Activation, type Encoding } from '../src/model/spec';

/** Minimal stand-in for the 2d canvas the trainer paints its output into. */
function stubCanvas(): HTMLCanvasElement {
  const ctx = {
    createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => {},
  };
  return { width: 0, height: 0, getContext: () => ctx } as unknown as HTMLCanvasElement;
}

function makeTarget(res: number): Float32Array {
  const t = new Float32Array(res * res * 4);
  for (let i = 0; i < res * res; i++) {
    const x = (i % res) / res;
    const y = Math.floor(i / res) / res;
    t[i * 4] = 0.5 + 0.4 * Math.sin(x * 7);
    t[i * 4 + 1] = 0.5 + 0.4 * Math.cos(y * 5);
    t[i * 4 + 2] = 0.5 + 0.3 * Math.sin((x + y) * 9);
    t[i * 4 + 3] = 1;
  }
  return t;
}

let failures = 0;

function gradientCheck(activation: Activation, encoding: Encoding) {
  const res = 8;
  const trainer = new CpuTrainer(
    stubCanvas(),
    {
      hiddenLayers: 2,
      width: 6,
      activation,
      encoding,
      frequencies: 3,
      encodingScale: 2,
      omega0: 4,
      seed: 7,
    },
    { ...DEFAULT_HYPER, lr: 0, batch: 24, weightDecay: 0 },
    8,
    8,
  );
  trainer.setImage(makeTarget(res), res, res);

  // Reach into the private state; this is a test harness, not production code.
  const t = trainer as unknown as {
    step(): number;
    params: Float32Array;
    grads: Float32Array;
    rngState: number;
  };

  const SEED = 20240917;
  const lossWithFixedBatch = () => {
    t.rngState = SEED;
    return t.step();
  };

  lossWithFixedBatch();
  const analytic = t.grads.slice();
  const n = analytic.length;

  /**
   * Perturbing one parameter at a time doesn't work here: parameters and
   * activations are float32 and the loss is averaged over batch*channels, so a
   * single component's contribution to the loss sits near the round-off floor.
   * Differencing along a whole direction instead raises the signal by orders of
   * magnitude, and validates every component in one shot.
   */
  // Steepest-descent direction: the highest-signal probe available.
  const norm = Math.hypot(...Array.from(analytic));

  /**
   * Error is scaled by the gradient norm rather than by the projection itself:
   * a random direction can land nearly orthogonal to the gradient, and dividing
   * by that near-zero projection would report a huge relative error for a
   * perfectly good derivative. Directions are unit length, so |g·d| <= ‖g‖ and
   * this stays well scaled.
   */
  const directional = (dir: Float32Array, eps: number) => {
    const base = t.params.slice();
    let analyticDot = 0;
    for (let i = 0; i < n; i++) analyticDot += analytic[i] * dir[i];

    for (let i = 0; i < n; i++) t.params[i] = base[i] + eps * dir[i];
    const up = lossWithFixedBatch();
    for (let i = 0; i < n; i++) t.params[i] = base[i] - eps * dir[i];
    const down = lossWithFixedBatch();
    t.params.set(base);

    const numeric = (up - down) / (2 * eps);
    return Math.abs(numeric - analyticDot) / norm;
  };

  const directions: Float32Array[] = [];
  const along = new Float32Array(n);
  for (let i = 0; i < n; i++) along[i] = analytic[i] / norm;
  directions.push(along);

  // Random directions catch components the gradient direction alone would miss.
  let seed = 991;
  const rand = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let k = 0; k < 5; k++) {
    const d = new Float32Array(n);
    let mag = 0;
    for (let i = 0; i < n; i++) {
      d[i] = rand() * 2 - 1;
      mag += d[i] * d[i];
    }
    mag = Math.sqrt(mag);
    for (let i = 0; i < n; i++) d[i] /= mag;
    directions.push(d);
  }

  let worst = 0;
  for (const d of directions) worst = Math.max(worst, directional(d, 1e-3));

  const ok = worst < 1e-3;
  if (!ok) failures++;
  console.log(
    `  ${ok ? 'pass' : 'FAIL'}  ${activation.padEnd(5)} + ${encoding.padEnd(10)}  ` +
      `max error ${worst.toExponential(2)} ‖g‖ over ${directions.length} directions`,
  );
}

function fitCheck(activation: Activation, encoding: Encoding, steps: number) {
  const res = 32;
  const trainer = new CpuTrainer(
    stubCanvas(),
    {
      hiddenLayers: 2,
      width: 32,
      activation,
      encoding,
      frequencies: 8,
      encodingScale: activation === 'sine' ? 1 : 4,
      omega0: 12,
      seed: 3,
    },
    { ...DEFAULT_HYPER, lr: 4e-3, batch: 256 },
    16,
    16,
  );
  trainer.setImage(makeTarget(res), res, res);

  trainer.tick(20, false);
  const start = trainer.loss();
  trainer.tick(steps, false);
  const end = trainer.loss();

  const ok = Number.isFinite(end) && end < start * 0.3;
  if (!ok) failures++;
  console.log(
    `  ${ok ? 'pass' : 'FAIL'}  ${activation.padEnd(5)} + ${encoding.padEnd(10)}  ` +
      `mse ${start.toExponential(2)} → ${end.toExponential(2)} after ${steps} steps`,
  );
}

console.log('\ngradient check (analytic vs. central difference)');
for (const a of ACTIVATIONS) {
  for (const e of ENCODINGS) gradientCheck(a, e);
}

console.log('\nfit check (loss must drop to under a third)');
for (const a of ACTIVATIONS) {
  fitCheck(a, a === 'sine' ? 'none' : 'gaussian', 3000);
}

console.log(failures === 0 ? '\nall checks passed\n' : `\n${failures} check(s) failed\n`);
process.exit(failures === 0 ? 0 : 1);
