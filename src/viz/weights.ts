import type { ModelLayout } from '../model/spec';
import { diverging, robustScale } from './color';
import { button, el } from '../ui/widgets';
import type { NeuronRef } from './architecture';

/**
 * Heatmap of one layer's weight matrix (plus its bias column), with direct
 * editing: click a cell, drag the slider, and the value is pushed back into
 * the live model.
 */

interface Cell {
  row: number;
  col: number;
  /** Index into the flat parameter array. */
  index: number;
  isBias: boolean;
}

const SEPARATOR_COLS = 1;

export class WeightsView {
  private root: HTMLElement;
  private tabsHost: HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private meta: HTMLElement;
  private editor: HTMLElement;
  private hint: HTMLElement;

  private layout: ModelLayout | null = null;
  private params: Float32Array | null = null;
  private layerIndex = 0;
  private cell: Cell | null = null;
  private highlightRow: number | null = null;
  private scale = 1;

  /** Called after an in-place edit so the trainer can re-upload the weights. */
  commit: ((p: Float32Array) => void) | null = null;
  onSelectNeuron: ((ref: NeuronRef | null) => void) | null = null;

  constructor(container: HTMLElement) {
    this.tabsHost = el('div', { class: 'layer-tabs' });
    this.canvas = el('canvas', { class: 'weight-map' });
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('could not acquire a 2d context for the weight map');
    this.ctx = ctx;

    this.meta = el('div', { class: 'weight-meta' });
    this.editor = el('div', { class: 'weight-editor' });
    this.hint = el('p', { class: 'muted' }, [
      'Rows are output units, columns are inputs; the detached column on the right is the bias. Click any cell to edit it.',
    ]);

    this.root = el('div', { class: 'weights-view' }, [
      this.tabsHost,
      el('div', { class: 'weight-map-wrap' }, [this.canvas]),
      this.meta,
      this.hint,
      this.editor,
    ]);
    container.replaceChildren(this.root);

    this.canvas.addEventListener('click', (e) => this.pick(e));
  }

  setLayout(layout: ModelLayout) {
    this.layout = layout;
    this.layerIndex = Math.min(this.layerIndex, layout.layers.length - 1);
    this.cell = null;
    this.buildTabs();
    this.resizeCanvas();
    this.renderEditor();
  }

  setSelection(ref: NeuronRef | null) {
    if (!this.layout) return;
    if (ref && ref.column > 0) {
      this.layerIndex = ref.column - 1;
      this.highlightRow = ref.neuron;
      this.buildTabs();
      this.resizeCanvas();
    } else {
      this.highlightRow = null;
    }
    this.draw();
  }

  private buildTabs() {
    if (!this.layout) return;
    const frag = document.createDocumentFragment();
    this.layout.layers.forEach((l) => {
      const b = button(
        l.isOutput ? `out ${l.inDim}→${l.outDim}` : `L${l.index} ${l.inDim}→${l.outDim}`,
        () => {
          this.layerIndex = l.index;
          this.cell = null;
          this.highlightRow = null;
          this.buildTabs();
          this.resizeCanvas();
          this.renderEditor();
        },
      );
      if (l.index === this.layerIndex) b.classList.add('on');
      frag.append(b);
    });
    this.tabsHost.replaceChildren(frag);
  }

  private get layer() {
    return this.layout?.layers[this.layerIndex] ?? null;
  }

  private resizeCanvas() {
    const l = this.layer;
    if (!l) return;
    this.canvas.width = l.inDim + SEPARATOR_COLS + 1;
    this.canvas.height = l.outDim;
    // Keep the map readable whether the matrix is 2×3 or 128×256.
    const aspect = this.canvas.width / this.canvas.height;
    this.canvas.style.aspectRatio = String(aspect);
    this.canvas.style.maxHeight = `${Math.min(340, Math.max(90, l.outDim * 14))}px`;
    this.draw();
  }

  update(params: Float32Array) {
    this.params = params;
    this.draw();
    this.syncEditorValue();
  }

  private draw() {
    const l = this.layer;
    const p = this.params;
    if (!l || !p) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    const img = this.ctx.createImageData(w, h);
    this.scale = robustScale(p, l.wOff, l.inDim * l.outDim);
    const biasScale = robustScale(p, l.bOff, l.outDim);

    for (let row = 0; row < h; row++) {
      const dim = this.highlightRow !== null && row !== this.highlightRow ? 0.35 : 1;
      for (let col = 0; col < w; col++) {
        const o = (row * w + col) * 4;
        let rgb: [number, number, number];
        if (col < l.inDim) {
          rgb = diverging(p[l.wOff + row * l.inDim + col] / this.scale);
        } else if (col < l.inDim + SEPARATOR_COLS) {
          rgb = [8, 9, 14];
        } else {
          rgb = diverging(p[l.bOff + row] / biasScale);
        }
        img.data[o] = rgb[0] * dim;
        img.data[o + 1] = rgb[1] * dim;
        img.data[o + 2] = rgb[2] * dim;
        img.data[o + 3] = 255;
      }
    }

    // Mark the selected cell in white so it stays findable at any zoom.
    if (this.cell && this.cell.row < h) {
      const col = this.cell.isBias ? l.inDim + SEPARATOR_COLS : this.cell.col;
      const o = (this.cell.row * w + col) * 4;
      img.data[o] = 255;
      img.data[o + 1] = 255;
      img.data[o + 2] = 255;
    }

    this.ctx.putImageData(img, 0, 0);

    const stats = layerStats(p, l);
    this.meta.replaceChildren(
      el('span', {}, [`${l.inDim * l.outDim + l.outDim} params`]),
      el('span', {}, [`σ ${stats.std.toFixed(3)}`]),
      el('span', {}, [`max |w| ${stats.max.toFixed(3)}`]),
      el('span', {}, [`dead rows ${stats.deadRows}`]),
    );
  }

  private pick(e: MouseEvent) {
    const l = this.layer;
    if (!l) return;
    const r = this.canvas.getBoundingClientRect();
    const col = Math.floor(((e.clientX - r.left) / r.width) * this.canvas.width);
    const row = Math.floor(((e.clientY - r.top) / r.height) * this.canvas.height);
    if (row < 0 || row >= l.outDim || col < 0) return;

    if (col < l.inDim) {
      this.cell = { row, col, index: l.wOff + row * l.inDim + col, isBias: false };
    } else if (col >= l.inDim + SEPARATOR_COLS) {
      this.cell = { row, col, index: l.bOff + row, isBias: true };
    } else {
      return;
    }
    this.highlightRow = row;
    this.onSelectNeuron?.({ column: this.layerIndex + 1, neuron: row });
    this.draw();
    this.renderEditor();
  }

  private valueInput: HTMLInputElement | null = null;
  private valueRange: HTMLInputElement | null = null;

  private renderEditor() {
    const l = this.layer;
    const p = this.params;
    this.valueInput = null;
    this.valueRange = null;

    if (!l || !p) {
      this.editor.replaceChildren();
      return;
    }

    const bulk = el('div', { class: 'weight-bulk' }, [
      button('Zero unit', () => this.mutate((q) => {
        const row = this.highlightRow ?? 0;
        for (let i = 0; i < l.inDim; i++) q[l.wOff + row * l.inDim + i] = 0;
        q[l.bOff + row] = 0;
      })),
      button('Scale ×0.5', () => this.scaleLayer(0.5)),
      button('Scale ×2', () => this.scaleLayer(2)),
      button('Prune small', () => this.mutate((q) => {
        const cut = this.scale * 0.15;
        for (let i = 0; i < l.inDim * l.outDim; i++) {
          if (Math.abs(q[l.wOff + i]) < cut) q[l.wOff + i] = 0;
        }
      })),
      button('Add noise', () => this.mutate((q) => {
        for (let i = 0; i < l.inDim * l.outDim; i++) {
          q[l.wOff + i] += (Math.random() * 2 - 1) * this.scale * 0.25;
        }
      })),
    ]);

    if (!this.cell) {
      this.editor.replaceChildren(
        el('p', { class: 'muted' }, ['No cell selected — layer-wide operations:']),
        bulk,
      );
      return;
    }

    const value = p[this.cell.index];
    const bound = Math.max(this.scale * 4, Math.abs(value) * 1.5, 0.05);

    const range = el('input', {
      type: 'range',
      min: String(-bound),
      max: String(bound),
      step: String(bound / 500),
      value: String(value),
    });
    const num = el('input', { type: 'number', step: '0.001', value: value.toFixed(4) });
    this.valueRange = range;
    this.valueInput = num;

    const apply = (v: number) => {
      if (!Number.isFinite(v)) return;
      this.mutate((q) => {
        q[this.cell!.index] = v;
      });
    };
    range.addEventListener('input', () => {
      num.value = Number(range.value).toFixed(4);
      apply(Number(range.value));
    });
    num.addEventListener('change', () => {
      range.value = num.value;
      apply(Number(num.value));
    });

    const label = this.cell.isBias
      ? `bias[${this.cell.row}]`
      : `W[${this.cell.row}, ${this.cell.col}]`;
    const from = this.cell.isBias
      ? '—'
      : this.layerIndex === 0
        ? (this.layout?.featureLabels[this.cell.col] ?? `feature ${this.cell.col}`)
        : `h${this.layerIndex} unit ${this.cell.col}`;

    this.editor.replaceChildren(
      el('div', { class: 'weight-cell-head' }, [
        el('strong', {}, [label]),
        el('span', { class: 'muted' }, [`from ${from}`]),
      ]),
      el('div', { class: 'weight-cell-edit' }, [range, num]),
      el('p', { class: 'muted' }, [
        'Edits apply immediately. Adam keeps its momentum, so training will pull the value back unless you pause first.',
      ]),
      bulk,
    );
  }

  /** Keep the open editor in sync while training moves the weight underneath it. */
  private syncEditorValue() {
    if (!this.cell || !this.params || !this.valueInput || !this.valueRange) return;
    if (document.activeElement === this.valueInput) return;
    const v = this.params[this.cell.index];
    this.valueInput.value = v.toFixed(4);
    this.valueRange.value = String(v);
  }

  private scaleLayer(k: number) {
    const l = this.layer;
    if (!l) return;
    this.mutate((q) => {
      for (let i = 0; i < l.inDim * l.outDim; i++) q[l.wOff + i] *= k;
      for (let j = 0; j < l.outDim; j++) q[l.bOff + j] *= k;
    });
  }

  private mutate(fn: (p: Float32Array) => void) {
    if (!this.params) return;
    fn(this.params);
    this.commit?.(this.params);
    this.draw();
  }
}

function layerStats(p: Float32Array, l: { wOff: number; bOff: number; inDim: number; outDim: number }) {
  let sum = 0;
  let sumSq = 0;
  let max = 0;
  const n = l.inDim * l.outDim;
  for (let i = 0; i < n; i++) {
    const v = p[l.wOff + i];
    sum += v;
    sumSq += v * v;
    if (Math.abs(v) > max) max = Math.abs(v);
  }
  const mean = sum / n;
  const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));

  // A row of near-zero weights means the unit contributes nothing downstream.
  let deadRows = 0;
  for (let j = 0; j < l.outDim; j++) {
    let rowMax = 0;
    for (let i = 0; i < l.inDim; i++) {
      rowMax = Math.max(rowMax, Math.abs(p[l.wOff + j * l.inDim + i]));
    }
    if (rowMax < std * 0.05) deadRows++;
  }
  return { std, max, deadRows };
}
