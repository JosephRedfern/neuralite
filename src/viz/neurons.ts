import type { ModelLayout } from '../model/spec';
import type { NeuronPreview } from '../model/trainer';
import { diverging, magma } from './color';
import { button, el } from '../ui/widgets';
import type { NeuronRef } from './architecture';

/**
 * Per-neuron spatial activation thumbnails: what each unit "looks at" across
 * the (x, y) domain. This is the view that makes a neural field legible —
 * early layers show the Fourier basis, later layers show composed structure.
 */

const THUMB_PX = 40;

export class NeuronsView {
  private root: HTMLElement;
  private tabsHost: HTMLElement;
  private grid: HTMLElement;
  private caption: HTMLElement;

  private layout: ModelLayout | null = null;
  private thumbs: HTMLCanvasElement[] = [];
  private layerIndex = 0;
  private selected: number | null = null;

  onSelectNeuron: ((ref: NeuronRef | null) => void) | null = null;
  onLayerChange: ((layerIndex: number) => void) | null = null;

  constructor(container: HTMLElement) {
    this.tabsHost = el('div', { class: 'layer-tabs' });
    this.grid = el('div', { class: 'neuron-grid' });
    this.caption = el('p', { class: 'muted' }, [
      'Each tile is one unit’s activation over the unit square, normalised independently.',
    ]);
    this.root = el('div', { class: 'neurons-view' }, [this.tabsHost, this.caption, this.grid]);
    container.replaceChildren(this.root);
  }

  setLayout(layout: ModelLayout) {
    this.layout = layout;
    this.layerIndex = Math.min(this.layerIndex, layout.layers.length - 1);
    this.thumbs = [];
    this.grid.replaceChildren();
    this.buildTabs();
  }

  setSelection(ref: NeuronRef | null) {
    if (ref && ref.column > 0 && ref.column - 1 === this.layerIndex) {
      this.selected = ref.neuron;
    } else {
      this.selected = null;
    }
    this.thumbs.forEach((c, i) => c.classList.toggle('selected', i === this.selected));
  }

  get activeLayer() {
    return this.layerIndex;
  }

  private buildTabs() {
    if (!this.layout) return;
    const frag = document.createDocumentFragment();
    this.layout.layers.forEach((l) => {
      const b = button(l.isOutput ? 'rgb' : `h${l.index + 1}`, () => {
        this.layerIndex = l.index;
        this.thumbs = [];
        this.grid.replaceChildren();
        this.buildTabs();
        this.onLayerChange?.(l.index);
      });
      if (l.index === this.layerIndex) b.classList.add('on');
      frag.append(b);
    });
    this.tabsHost.replaceChildren(frag);
  }

  update(preview: NeuronPreview) {
    if (!this.layout) return;
    const layer = this.layout.layers[this.layerIndex];
    if (!layer || preview.count !== layer.outDim) return;

    if (this.thumbs.length !== preview.count) this.buildGrid(preview);

    const { width, height, data, stride, count } = preview;
    const signed = !layer.isOutput && (this.layout.spec.activation === 'tanh' ||
      this.layout.spec.activation === 'sine');

    for (let j = 0; j < count; j++) {
      const canvas = this.thumbs[j];
      const ctx = canvas.getContext('2d');
      if (!ctx) continue;

      let peak = 1e-6;
      for (let p = 0; p < width * height; p++) {
        const v = Math.abs(data[p * stride + j]);
        if (v > peak) peak = v;
      }

      const img = ctx.createImageData(width, height);
      for (let p = 0; p < width * height; p++) {
        const v = data[p * stride + j] / peak;
        const rgb = signed ? diverging(v) : magma(layer.isOutput ? data[p * stride + j] : Math.abs(v));
        const o = p * 4;
        img.data[o] = rgb[0];
        img.data[o + 1] = rgb[1];
        img.data[o + 2] = rgb[2];
        img.data[o + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      canvas.title = `unit ${j} · peak ${peak.toFixed(3)}`;
    }
  }

  private buildGrid(preview: NeuronPreview) {
    const frag = document.createDocumentFragment();
    this.thumbs = [];
    for (let j = 0; j < preview.count; j++) {
      const c = el('canvas', { class: 'neuron-thumb' });
      c.width = preview.width;
      c.height = preview.height;
      c.style.width = `${THUMB_PX}px`;
      c.style.height = `${THUMB_PX}px`;
      c.addEventListener('click', () => {
        this.selected = this.selected === j ? null : j;
        this.thumbs.forEach((t, i) => t.classList.toggle('selected', i === this.selected));
        this.onSelectNeuron?.(
          this.selected === null ? null : { column: this.layerIndex + 1, neuron: j },
        );
      });
      this.thumbs.push(c);
      frag.append(c);
    }
    this.grid.replaceChildren(frag);
  }
}
