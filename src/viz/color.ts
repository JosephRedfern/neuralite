/**
 * Diverging blue → grey → orange ramp, used everywhere a signed quantity
 * (weight, activation) is drawn. `t` is expected in [-1, 1].
 */
export function diverging(t: number): [number, number, number] {
  const c = Math.max(-1, Math.min(1, t));
  const neg: [number, number, number] = [70, 150, 235];
  const mid: [number, number, number] = [26, 28, 38];
  const pos: [number, number, number] = [255, 138, 76];
  const from = c < 0 ? neg : pos;
  const k = Math.abs(c);
  // Ease so small magnitudes stay dark and readable against the panel.
  const e = Math.pow(k, 0.7);
  return [
    Math.round(mid[0] + (from[0] - mid[0]) * e),
    Math.round(mid[1] + (from[1] - mid[1]) * e),
    Math.round(mid[2] + (from[2] - mid[2]) * e),
  ];
}

export function divergingCss(t: number, alpha = 1): string {
  const [r, g, b] = diverging(t);
  return alpha >= 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
}

/** Sequential ramp for unsigned magnitudes (e.g. relu activations). */
export function magma(t: number): [number, number, number] {
  const k = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [
    [12, 12, 24],
    [70, 30, 100],
    [180, 54, 110],
    [244, 130, 78],
    [252, 236, 190],
  ];
  const s = k * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(s));
  const f = s - i;
  return [
    Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f),
    Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f),
    Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f),
  ];
}

/** Robust scale for a weight/activation slab: the 99th percentile magnitude. */
export function robustScale(values: Float32Array, from = 0, count = values.length): number {
  const n = Math.min(count, values.length - from);
  if (n <= 0) return 1;
  const stride = Math.max(1, Math.floor(n / 4096));
  const sample: number[] = [];
  for (let i = 0; i < n; i += stride) sample.push(Math.abs(values[from + i]));
  sample.sort((a, b) => a - b);
  const v = sample[Math.floor(sample.length * 0.99)] ?? 1;
  return v > 1e-6 ? v : 1;
}
