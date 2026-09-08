import type { ModelLayout } from './spec';

export interface HyperParams {
  lr: number;
  batch: number;
  weightDecay: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export const DEFAULT_HYPER: HyperParams = {
  lr: 3e-3,
  batch: 4096,
  weightDecay: 0,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
};

/** A view window over the field, in normalised coordinates. 1 = the trained unit square. */
export interface ViewBox {
  zoom: number;
  cx: number;
  cy: number;
}

export interface NeuronPreview {
  width: number;
  height: number;
  /** Row-major [pixel][stride]; neuron j of pixel p is data[p * stride + j]. */
  data: Float32Array;
  stride: number;
  count: number;
}

/**
 * Both backends implement this. `tick` enqueues a frame's worth of work;
 * everything else is small and synchronous apart from GPU readback.
 */
export interface Trainer {
  readonly backend: 'webgpu' | 'cpu';
  readonly layout: ModelLayout;
  readonly canvas: HTMLCanvasElement;
  readonly stepCount: number;

  /** Run `steps` optimiser steps, then optionally repaint the field canvas. */
  tick(steps: number, render: boolean): void;

  /** Mean squared error over the most recent measured batch. */
  loss(): number;

  /** Target image as rgba floats in [0,1], row-major, stride 4. */
  setImage(data: Float32Array, width: number, height: number): void;
  setHyper(h: HyperParams): void;
  setView(view: ViewBox): void;
  setOutputSize(width: number, height: number): void;

  /** CPU-side mirror of the weights. On WebGPU this lags until `refreshParams`. */
  getParams(): Float32Array;
  /** Overwrite the weights (used by the weight editor). */
  setParams(p: Float32Array): void;
  /** Pull the live weights back from the device. */
  refreshParams(): Promise<Float32Array>;

  /** Spatial activation map for every neuron in `layerIndex`'s output. */
  neuronPreview(layerIndex: number, size: number): Promise<NeuronPreview | null>;

  destroy(): void;
}
