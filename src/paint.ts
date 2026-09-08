/**
 * The left-hand target canvas: freehand painting, image upload, and conversion
 * to the float rgba buffer the trainer samples from.
 */

export const PAINT_RES = 512;

export type Tool = 'brush' | 'eraser' | 'fill';

export interface PaintOptions {
  color: string;
  size: number;
  tool: Tool;
  softness: number;
}

export class PaintCanvas {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private scratch: HTMLCanvasElement;
  private scratchCtx: CanvasRenderingContext2D;

  private drawing = false;
  private last: { x: number; y: number } | null = null;
  private undoStack: ImageData[] = [];
  private redoStack: ImageData[] = [];
  private dirty = true;

  opts: PaintOptions = { color: '#ff5c8a', size: 48, tool: 'brush', softness: 0.35 };
  onChange: (() => void) | null = null;

  constructor(container: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = PAINT_RES;
    this.canvas.height = PAINT_RES;
    this.canvas.className = 'paint-canvas';
    container.appendChild(this.canvas);

    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('could not acquire a 2d context for the paint canvas');
    this.ctx = ctx;

    this.scratch = document.createElement('canvas');
    const sctx = this.scratch.getContext('2d', { willReadFrequently: true });
    if (!sctx) throw new Error('could not acquire a 2d context for downscaling');
    this.scratchCtx = sctx;

    this.clear('#12141c');
    this.attach();
  }

  private attach() {
    const pos = (e: PointerEvent) => {
      const r = this.canvas.getBoundingClientRect();
      return {
        x: ((e.clientX - r.left) / r.width) * PAINT_RES,
        y: ((e.clientY - r.top) / r.height) * PAINT_RES,
      };
    };

    this.canvas.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      this.canvas.setPointerCapture(e.pointerId);
      this.pushUndo();
      const p = pos(e);
      if (this.opts.tool === 'fill') {
        this.clear(this.opts.color);
        return;
      }
      this.drawing = true;
      this.last = p;
      this.dab(p.x, p.y);
      this.notify();
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.drawing) return;
      const p = pos(e);
      // Coalesced events keep fast strokes smooth on high-rate pointers.
      const points = e.getCoalescedEvents ? e.getCoalescedEvents().map(pos) : [p];
      for (const q of points) {
        if (this.last) this.stroke(this.last, q);
        this.last = q;
      }
      this.notify();
    });

    const end = () => {
      if (!this.drawing) return;
      this.drawing = false;
      this.last = null;
      this.notify();
    };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
    this.canvas.addEventListener('pointerleave', end);
  }

  private brushStyle(x: number, y: number) {
    const r = this.opts.size / 2;
    const color = this.opts.tool === 'eraser' ? '#12141c' : this.opts.color;
    if (this.opts.softness <= 0.01) return color;
    const g = this.ctx.createRadialGradient(x, y, r * (1 - this.opts.softness), x, y, r);
    g.addColorStop(0, color);
    g.addColorStop(1, hexToRgba(color, 0));
    return g;
  }

  private dab(x: number, y: number) {
    this.ctx.fillStyle = this.brushStyle(x, y);
    this.ctx.beginPath();
    this.ctx.arc(x, y, this.opts.size / 2, 0, Math.PI * 2);
    this.ctx.fill();
    this.dirty = true;
  }

  private stroke(a: { x: number; y: number }, b: { x: number; y: number }) {
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    const stepSize = Math.max(1, this.opts.size * 0.12);
    const steps = Math.max(1, Math.ceil(dist / stepSize));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      this.dab(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    }
  }

  private notify() {
    this.onChange?.();
  }

  private pushUndo() {
    this.undoStack.push(this.ctx.getImageData(0, 0, PAINT_RES, PAINT_RES));
    if (this.undoStack.length > 24) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  undo() {
    const prev = this.undoStack.pop();
    if (!prev) return;
    this.redoStack.push(this.ctx.getImageData(0, 0, PAINT_RES, PAINT_RES));
    this.ctx.putImageData(prev, 0, 0);
    this.dirty = true;
    this.notify();
  }

  redo() {
    const next = this.redoStack.pop();
    if (!next) return;
    this.undoStack.push(this.ctx.getImageData(0, 0, PAINT_RES, PAINT_RES));
    this.ctx.putImageData(next, 0, 0);
    this.dirty = true;
    this.notify();
  }

  clear(color: string) {
    this.ctx.globalCompositeOperation = 'source-over';
    this.ctx.fillStyle = color;
    this.ctx.fillRect(0, 0, PAINT_RES, PAINT_RES);
    this.dirty = true;
    this.notify();
  }

  clearWithUndo(color: string) {
    this.pushUndo();
    this.clear(color);
  }

  /** Draw `src` centred and cover-cropped into the square canvas. */
  drawImage(src: CanvasImageSource, w: number, h: number) {
    this.pushUndo();
    const scale = Math.max(PAINT_RES / w, PAINT_RES / h);
    const dw = w * scale;
    const dh = h * scale;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.ctx.fillStyle = '#12141c';
    this.ctx.fillRect(0, 0, PAINT_RES, PAINT_RES);
    this.ctx.drawImage(src, (PAINT_RES - dw) / 2, (PAINT_RES - dh) / 2, dw, dh);
    this.dirty = true;
    this.notify();
  }

  async loadFile(file: File) {
    const bitmap = await createImageBitmap(file);
    try {
      this.drawImage(bitmap, bitmap.width, bitmap.height);
    } finally {
      bitmap.close();
    }
  }

  drawSample(kind: 'gradient' | 'rings' | 'checker' | 'blobs') {
    this.pushUndo();
    const c = this.ctx;
    const N = PAINT_RES;
    const img = c.createImageData(N, N);
    for (let py = 0; py < N; py++) {
      for (let px = 0; px < N; px++) {
        const u = px / N;
        const v = py / N;
        const x = u * 2 - 1;
        const y = v * 2 - 1;
        let r = 0;
        let g = 0;
        let b = 0;
        if (kind === 'gradient') {
          r = u;
          g = v;
          b = 0.5 + 0.5 * Math.sin((u + v) * 3);
        } else if (kind === 'rings') {
          const d = Math.hypot(x, y);
          const t = 0.5 + 0.5 * Math.sin(d * 18);
          r = t;
          g = 0.5 + 0.5 * Math.sin(d * 18 + 2);
          b = 0.5 + 0.5 * Math.sin(d * 18 + 4);
        } else if (kind === 'checker') {
          const s = (Math.floor(u * 8) + Math.floor(v * 8)) % 2;
          r = s ? 0.95 : 0.08;
          g = s ? 0.36 : 0.1;
          b = s ? 0.54 : 0.16;
        } else {
          const f =
            Math.sin(x * 3.1) * Math.cos(y * 2.3) + 0.6 * Math.sin(x * 6.7 + y * 5.1);
          r = 0.5 + 0.5 * Math.sin(f * 2);
          g = 0.5 + 0.5 * Math.sin(f * 2 + 2.1);
          b = 0.5 + 0.5 * Math.sin(f * 2 + 4.2);
        }
        const o = (py * N + px) * 4;
        img.data[o] = r * 255;
        img.data[o + 1] = g * 255;
        img.data[o + 2] = b * 255;
        img.data[o + 3] = 255;
      }
    }
    c.putImageData(img, 0, 0);
    this.dirty = true;
    this.notify();
  }

  get isDirty() {
    return this.dirty;
  }

  /** Box-downsample the full painting onto the shared scratch canvas at `res`. */
  private downsample(res: number): HTMLCanvasElement {
    this.scratch.width = res;
    this.scratch.height = res;
    this.scratchCtx.imageSmoothingEnabled = true;
    this.scratchCtx.imageSmoothingQuality = 'high';
    this.scratchCtx.clearRect(0, 0, res, res);
    this.scratchCtx.drawImage(this.canvas, 0, 0, res, res);
    return this.scratch;
  }

  /**
   * Box-downsample to `res` and return rgba floats in [0,1] (stride 4), which
   * is the layout both trainers expect for the target image.
   */
  toTarget(res: number): Float32Array {
    this.downsample(res);
    const src = this.scratchCtx.getImageData(0, 0, res, res).data;
    const out = new Float32Array(res * res * 4);
    for (let i = 0; i < res * res * 4; i++) out[i] = src[i] / 255;
    this.dirty = false;
    return out;
  }

  /**
   * PNG-encoded size of the painting at `res`, in bytes — used as a real-world
   * "conventional compression" baseline to compare the model's parameter
   * count against.
   */
  toPngBytes(res: number): Promise<number> {
    const canvas = this.downsample(res);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob ? blob.size : 0), 'image/png');
    });
  }

  /** JPEG-encoded size of the painting at `res`, in bytes, at the given quality (0-1). */
  toJpegBytes(res: number, quality: number): Promise<number> {
    const canvas = this.downsample(res);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob ? blob.size : 0), 'image/jpeg', quality);
    });
  }
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
