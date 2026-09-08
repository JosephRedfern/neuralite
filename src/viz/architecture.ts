import type { ModelLayout } from '../model/spec';
import { divergingCss, robustScale } from './color';
import { svgEl } from '../ui/widgets';

/**
 * Column-per-layer diagram of the MLP. Nodes are neurons, edges are weights
 * coloured by sign and magnitude. Columns are indexed 0 = encoded input,
 * 1..L = each weight layer's output.
 */

const VB_W = 1000;
const VB_H = 430;
const MAX_NODES = 16;

export interface NeuronRef {
  column: number;
  neuron: number;
}

interface NodeView {
  column: number;
  neuron: number;
  x: number;
  y: number;
  circle: SVGElement;
}

interface EdgeView {
  line: SVGElement;
  paramIndex: number;
  layerIndex: number;
  fromNode: NodeView;
  toNode: NodeView;
}

export class ArchitectureView {
  private svg: SVGElement;
  private edgeGroup: SVGElement;
  private nodeGroup: SVGElement;
  private labelGroup: SVGElement;
  private tooltip: HTMLElement;

  private nodes: NodeView[] = [];
  private edges: EdgeView[] = [];
  private layout: ModelLayout | null = null;
  private selection: NeuronRef | null = null;
  private hover: NodeView | null = null;

  onSelect: ((ref: NeuronRef | null) => void) | null = null;

  constructor(private container: HTMLElement) {
    this.svg = svgEl('svg', {
      viewBox: `0 0 ${VB_W} ${VB_H}`,
      preserveAspectRatio: 'xMidYMid meet',
      class: 'arch-svg',
    });
    this.edgeGroup = svgEl('g', { class: 'edges' });
    this.nodeGroup = svgEl('g', { class: 'nodes' });
    this.labelGroup = svgEl('g', { class: 'labels' });
    this.svg.append(this.edgeGroup, this.nodeGroup, this.labelGroup);

    this.tooltip = document.createElement('div');
    this.tooltip.className = 'arch-tip';
    this.tooltip.hidden = true;

    container.replaceChildren(this.svg, this.tooltip);

    this.svg.addEventListener('pointerleave', () => {
      this.hover = null;
      this.tooltip.hidden = true;
      this.applyEmphasis();
    });
  }

  /** Rebuild the diagram for a new architecture. */
  setLayout(layout: ModelLayout) {
    this.layout = layout;
    this.selection = null;
    this.build();
  }

  setSelection(ref: NeuronRef | null) {
    this.selection = ref;
    this.applyEmphasis();
  }

  private columnSizes(): number[] {
    const l = this.layout!;
    return [l.inputDim, ...l.layers.map((x) => x.outDim)];
  }

  /** Indices actually drawn for a column of `n` neurons, with a gap marker. */
  private visibleIndices(n: number): (number | null)[] {
    if (n <= MAX_NODES) return Array.from({ length: n }, (_, i) => i);
    const head = Math.ceil(MAX_NODES / 2);
    const tail = MAX_NODES - head;
    const out: (number | null)[] = [];
    for (let i = 0; i < head; i++) out.push(i);
    out.push(null);
    for (let i = n - tail; i < n; i++) out.push(i);
    return out;
  }

  private build() {
    const layout = this.layout!;
    const sizes = this.columnSizes();
    this.nodes = [];
    this.edges = [];
    const edgeFrag = document.createDocumentFragment();
    const nodeFrag = document.createDocumentFragment();
    const labelFrag = document.createDocumentFragment();

    const padX = 70;
    const spanX = VB_W - padX * 2;
    const colX = (i: number) =>
      sizes.length === 1 ? VB_W / 2 : padX + (spanX * i) / (sizes.length - 1);

    const perColumn: NodeView[][] = [];

    sizes.forEach((n, col) => {
      const vis = this.visibleIndices(n);
      const top = 56;
      const bottom = VB_H - 74;
      const step = (bottom - top) / Math.max(1, vis.length - 1);
      const x = colX(col);
      const nodes: NodeView[] = [];
      const r = n > MAX_NODES ? 6 : Math.min(10, 5 + 40 / Math.max(4, n));

      vis.forEach((idx, slot) => {
        const y = vis.length === 1 ? (top + bottom) / 2 : top + slot * step;
        if (idx === null) {
          const dots = svgEl('text', {
            x,
            y: y + 4,
            class: 'arch-ellipsis',
            'text-anchor': 'middle',
          });
          dots.textContent = `⋯ ${n - MAX_NODES} more ⋯`;
          labelFrag.append(dots);
          return;
        }
        const circle = svgEl('circle', { cx: x, cy: y, r, class: 'arch-node' });
        const node: NodeView = { column: col, neuron: idx, x, y, circle };
        circle.addEventListener('pointerenter', (e) => this.onNodeHover(node, e as PointerEvent));
        circle.addEventListener('click', () => this.onNodeClick(node));
        nodes.push(node);
        this.nodes.push(node);
        nodeFrag.append(circle);
      });

      perColumn.push(nodes);

      // Column caption.
      const title = svgEl('text', { x, y: 30, class: 'arch-col-title', 'text-anchor': 'middle' });
      title.textContent = col === 0 ? 'input' : col === sizes.length - 1 ? 'rgb' : `h${col}`;
      const sub = svgEl('text', { x, y: 46, class: 'arch-col-sub', 'text-anchor': 'middle' });
      sub.textContent = `${n}`;
      labelFrag.append(title, sub);

      if (col > 0) {
        const l = layout.layers[col - 1];
        const meta = svgEl('text', {
          x: (colX(col - 1) + x) / 2,
          y: VB_H - 34,
          class: 'arch-edge-label',
          'text-anchor': 'middle',
        });
        meta.textContent = `${l.inDim}×${l.outDim} + ${l.outDim}`;
        const act = svgEl('text', {
          x: (colX(col - 1) + x) / 2,
          y: VB_H - 18,
          class: 'arch-edge-act',
          'text-anchor': 'middle',
        });
        act.textContent = l.isOutput ? 'sigmoid' : layout.spec.activation;
        labelFrag.append(meta, act);
      }
    });

    // Edges between consecutive drawn columns.
    for (let col = 1; col < perColumn.length; col++) {
      const l = layout.layers[col - 1];
      for (const to of perColumn[col]) {
        for (const from of perColumn[col - 1]) {
          const line = svgEl('line', {
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            class: 'arch-edge',
          });
          const edge: EdgeView = {
            line,
            paramIndex: l.wOff + to.neuron * l.inDim + from.neuron,
            layerIndex: l.index,
            fromNode: from,
            toNode: to,
          };
          line.addEventListener('pointerenter', (e) => this.onEdgeHover(edge, e as PointerEvent));
          this.edges.push(edge);
          edgeFrag.append(line);
        }
      }
    }

    this.edgeGroup.replaceChildren(edgeFrag);
    this.nodeGroup.replaceChildren(nodeFrag);
    this.labelGroup.replaceChildren(labelFrag);
  }

  /** Recolour edges/nodes from the current weights. Cheap enough to run per frame. */
  update(params: Float32Array) {
    if (!this.layout) return;
    this.lastParams = params;
    const scales = this.layout.layers.map((l) =>
      robustScale(params, l.wOff, l.inDim * l.outDim),
    );

    for (const e of this.edges) {
      const w = params[e.paramIndex];
      const t = w / scales[e.layerIndex];
      const mag = Math.min(1, Math.abs(t));
      e.line.setAttribute('stroke', divergingCss(t));
      e.line.setAttribute('stroke-width', String(0.4 + mag * 2.2));
      e.line.setAttribute('data-op', String(0.1 + mag * 0.75));
    }

    // Node fill tracks its bias, so dead/saturated units read at a glance.
    for (const n of this.nodes) {
      if (n.column === 0) {
        n.circle.setAttribute('fill', '#4a5570');
        continue;
      }
      const l = this.layout.layers[n.column - 1];
      const b = params[l.bOff + n.neuron];
      n.circle.setAttribute('fill', divergingCss(b / (scales[l.index] * 2)));
    }

    this.applyEmphasis();
  }

  private applyEmphasis() {
    const focus = this.hover ?? this.toNodeView(this.selection);
    for (const e of this.edges) {
      const base = Number(e.line.getAttribute('data-op') ?? 0.4);
      if (!focus) {
        e.line.setAttribute('opacity', String(base));
        continue;
      }
      const touches = e.fromNode === focus || e.toNode === focus;
      e.line.setAttribute('opacity', String(touches ? Math.max(0.85, base) : base * 0.12));
    }
    for (const n of this.nodes) {
      const sel = this.selection && n.column === this.selection.column && n.neuron === this.selection.neuron;
      n.circle.classList.toggle('selected', Boolean(sel));
      n.circle.classList.toggle('hovered', n === focus);
    }
  }

  private toNodeView(ref: NeuronRef | null): NodeView | null {
    if (!ref) return null;
    return this.nodes.find((n) => n.column === ref.column && n.neuron === ref.neuron) ?? null;
  }

  private onNodeHover(node: NodeView, e: PointerEvent) {
    this.hover = node;
    const layout = this.layout!;
    const label =
      node.column === 0
        ? layout.featureLabels[node.neuron] ?? `feature ${node.neuron}`
        : node.column === layout.layers.length
          ? `output ${'rgb'[node.neuron] ?? node.neuron}`
          : `h${node.column} unit ${node.neuron}`;
    this.showTip(`${label} — click to inspect`, e);
    this.applyEmphasis();
  }

  private onEdgeHover(edge: EdgeView, e: PointerEvent) {
    const params = this.lastParams;
    const w = params ? params[edge.paramIndex] : NaN;
    this.showTip(
      `w[${edge.toNode.neuron}, ${edge.fromNode.neuron}] layer ${edge.layerIndex} = ${w.toFixed(4)}`,
      e,
    );
  }

  /** Kept so edge tooltips can report live values without another readback. */
  private lastParams: Float32Array | null = null;

  private showTip(text: string, e: PointerEvent) {
    const rect = this.container.getBoundingClientRect();
    this.tooltip.textContent = text;
    this.tooltip.hidden = false;
    this.tooltip.style.left = `${e.clientX - rect.left + 12}px`;
    this.tooltip.style.top = `${e.clientY - rect.top + 12}px`;
  }

  private onNodeClick(node: NodeView) {
    const same =
      this.selection?.column === node.column && this.selection?.neuron === node.neuron;
    this.selection = same ? null : { column: node.column, neuron: node.neuron };
    this.applyEmphasis();
    this.onSelect?.(this.selection);
  }
}
