import './style.css';

import { CpuTrainer } from './model/cpu';
import { GpuTrainer, requestDevice } from './model/gpu';
import {
  ACTIVATIONS,
  DEFAULT_SPEC,
  ENCODINGS,
  buildLayout,
  describeSpec,
  type Activation,
  type Encoding,
  type ModelSpec,
} from './model/spec';
import { DEFAULT_HYPER, type HyperParams, type Trainer, type ViewBox } from './model/trainer';
import { PaintCanvas } from './paint';
import { ArchitectureView, type NeuronRef } from './viz/architecture';
import { NeuronsView } from './viz/neurons';
import { WeightsView } from './viz/weights';
import { button, el, fieldset, kvRow, kvSeparator, segmented, select, slider } from './ui/widgets';
import { formatBytes, formatRatio } from './ui/format';

const PARAM_REFRESH_MS = 120;
const NEURON_REFRESH_MS = 450;
const TARGET_UPLOAD_MS = 90;
const PREVIEW_GRID = 32;

interface AppState {
  spec: ModelSpec;
  hyper: HyperParams;
  view: ViewBox;
  trainRes: number;
  outRes: number;
  stepsPerFrame: number;
  running: boolean;
}

const SWATCHES = [
  '#ff5c8a', '#ffd166', '#4ade80', '#38bdf8',
  '#a78bfa', '#f97316', '#f1f5f9', '#12141c',
];

async function boot() {
  const device = await requestDevice();
  const gpu = device !== null;

  const state: AppState = {
    spec: { ...DEFAULT_SPEC },
    hyper: { ...DEFAULT_HYPER },
    view: { zoom: 1, cx: 0, cy: 0 },
    trainRes: 128,
    outRes: gpu ? 256 : 128,
    stepsPerFrame: gpu ? 12 : 1,
    running: true,
  };

  // The CPU path runs the same maths in scalar JS, so it needs a smaller model
  // and batch to stay interactive.
  if (!gpu) {
    state.spec.hiddenLayers = 2;
    state.spec.width = 32;
    state.spec.frequencies = 12;
    state.hyper.batch = 512;
  }

  const paintWrap = must('paint-wrap');
  const fieldWrap = must('field-wrap');

  const paint = new PaintCanvas(paintWrap);
  paint.drawSample('blobs');

  const fieldCanvas = el('canvas', { class: 'field-canvas' });
  fieldWrap.append(fieldCanvas);

  let trainer: Trainer = makeTrainer();
  function makeTrainer(): Trainer {
    if (device) {
      return GpuTrainer.create(
        device,
        fieldCanvas,
        state.spec,
        state.hyper,
        state.outRes,
        state.outRes,
      );
    }
    return new CpuTrainer(fieldCanvas, state.spec, state.hyper, state.outRes, state.outRes);
  }

  const weightsHost = el('div', { class: 'panel-body-inner' });
  const neuronsHost = el('div', { class: 'panel-body-inner' });

  const arch = new ArchitectureView(must('arch'));
  const weights = new WeightsView(weightsHost);
  const neurons = new NeuronsView(neuronsHost);

  const setSelection = (ref: NeuronRef | null) => {
    arch.setSelection(ref);
    weights.setSelection(ref);
    neurons.setSelection(ref);
  };
  arch.onSelect = setSelection;
  weights.onSelectNeuron = setSelection;
  neurons.onSelectNeuron = setSelection;
  weights.commit = (p) => trainer.setParams(p);

  // ------------------------------------------------------------ inspector tabs

  const inspectHost = must('inspect');
  inspectHost.replaceChildren(weightsHost);

  let inspectTab: 'weights' | 'neurons' = 'weights';
  const showTab = (tab: 'weights' | 'neurons') => {
    inspectTab = tab;
    inspectHost.replaceChildren(tab === 'weights' ? weightsHost : neuronsHost);
  };
  must('inspect-tabs').append(
    segmented(
      [
        { value: 'weights', label: 'Weights' },
        { value: 'neurons', label: 'Neurons' },
      ] as const,
      'weights',
      showTab,
    ),
  );

  // ------------------------------------------------------------ paint tools

  const swatchRow = el('div', { class: 'swatches' });
  const colorInput = el('input', { type: 'color', value: paint.opts.color, class: 'color-input' });
  colorInput.addEventListener('input', () => {
    paint.opts.color = colorInput.value;
    markSwatch(colorInput.value);
  });
  const swatchButtons = SWATCHES.map((c) => {
    const b = el('button', { class: 'swatch', style: `--sw:${c}`, title: c });
    b.addEventListener('click', () => {
      paint.opts.color = c;
      colorInput.value = c;
      markSwatch(c);
    });
    return b;
  });
  function markSwatch(color: string) {
    swatchButtons.forEach((b, i) => b.classList.toggle('on', SWATCHES[i] === color));
  }
  markSwatch(paint.opts.color);
  swatchRow.append(...swatchButtons, colorInput);

  const fileInput = el('input', { type: 'file', accept: 'image/*', class: 'hidden-file' });
  fileInput.addEventListener('change', async () => {
    const f = fileInput.files?.[0];
    if (f) await paint.loadFile(f);
    fileInput.value = '';
  });

  must('paint-tools').append(
    segmented(
      [
        { value: 'brush', label: 'Brush' },
        { value: 'eraser', label: 'Erase' },
        { value: 'fill', label: 'Fill' },
      ] as const,
      'brush',
      (t) => (paint.opts.tool = t),
    ),
    button('Upload', () => fileInput.click()),
    fileInput,
  );

  const brushSize = slider({
    label: 'Brush',
    min: 4,
    max: 220,
    step: 1,
    value: paint.opts.size,
    format: (v) => `${v}px`,
    onInput: (v) => (paint.opts.size = v),
  });
  const brushSoft = slider({
    label: 'Softness',
    min: 0,
    max: 1,
    step: 0.01,
    value: paint.opts.softness,
    format: (v) => v.toFixed(2),
    onInput: (v) => (paint.opts.softness = v),
  });

  must('paint-foot').append(
    swatchRow,
    el('div', { class: 'row' }, [brushSize.root, brushSoft.root]),
    el('div', { class: 'row wrap' }, [
      button('Clear', () => paint.clearWithUndo('#12141c')),
      button('Undo', () => paint.undo()),
      button('Redo', () => paint.redo()),
      button('Gradient', () => paint.drawSample('gradient')),
      button('Rings', () => paint.drawSample('rings')),
      button('Checker', () => paint.drawSample('checker')),
      button('Blobs', () => paint.drawSample('blobs')),
    ]),
  );

  // ------------------------------------------------------------ field tools

  must('field-tools').append(
    button('Reset view', () => {
      state.view = { zoom: 1, cx: 0, cy: 0 };
      trainer.setView(state.view);
      zoomCtl.set(1);
    }),
    button('PNG', () => downloadCanvas(fieldCanvas)),
  );

  // Drag to pan, wheel to zoom — handy for seeing how the field extrapolates
  // outside the square it was trained on.
  let panning: { x: number; y: number } | null = null;
  fieldCanvas.addEventListener('pointerdown', (e) => {
    fieldCanvas.setPointerCapture(e.pointerId);
    panning = { x: e.clientX, y: e.clientY };
  });
  fieldCanvas.addEventListener('pointermove', (e) => {
    if (!panning) return;
    const r = fieldCanvas.getBoundingClientRect();
    state.view.cx -= ((e.clientX - panning.x) / r.width) * 2 * state.view.zoom;
    state.view.cy -= ((e.clientY - panning.y) / r.height) * 2 * state.view.zoom;
    panning = { x: e.clientX, y: e.clientY };
    trainer.setView(state.view);
  });
  const endPan = () => (panning = null);
  fieldCanvas.addEventListener('pointerup', endPan);
  fieldCanvas.addEventListener('pointercancel', endPan);
  fieldCanvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      state.view.zoom = clamp(state.view.zoom * Math.exp(e.deltaY * 0.0015), 0.2, 8);
      trainer.setView(state.view);
      zoomCtl.set(state.view.zoom);
    },
    { passive: false },
  );

  // ------------------------------------------------------------ controls

  let rebuildQueued = false;
  function rebuildModel() {
    if (rebuildQueued) return;
    rebuildQueued = true;
    // Coalesce rapid slider drags into a single expensive rebuild.
    requestAnimationFrame(() => {
      rebuildQueued = false;
      const old = trainer;
      trainer = makeTrainer();
      old.destroy();
      trainer.setImage(lastTarget, state.trainRes, state.trainRes);
      trainer.setView(state.view);
      lossHistory.length = 0;
      onLayoutChanged();
    });
  }

  function onLayoutChanged() {
    arch.setLayout(trainer.layout);
    weights.setLayout(trainer.layout);
    neurons.setLayout(trainer.layout);
    setSelection(null);
    text('stat-params', trainer.layout.paramCount.toLocaleString());
    shapeLabel.textContent = describeSpec(trainer.layout);
    updateSizeStats();
  }

  const shapeLabel = el('code', { class: 'shape' }, ['']);

  const sizeModelFp32 = el('span', { class: 'kv-value' }, ['—']);
  const sizeModelFp16 = el('span', { class: 'kv-value dim' }, ['—']);
  const sizeModelInt8 = el('span', { class: 'kv-value dim' }, ['—']);
  const sizeTargetRaw = el('span', { class: 'kv-value' }, ['—']);
  const sizeTargetPng = el('span', { class: 'kv-value' }, ['—']);
  const sizeTargetJpeg = el('span', { class: 'kv-value' }, ['—']);
  const sizeRatioPng = el('span', { class: 'kv-value accent' }, ['—']);
  const sizeRatioJpeg = el('span', { class: 'kv-value accent' }, ['—']);

  const JPEG_QUALITY = 0.8;
  let targetPngBytes = 0;
  let targetJpegBytes = 0;
  let compressionMeasurePending = false;

  function updateSizeStats() {
    const n = trainer.layout.paramCount;
    const modelBytes = n * 4;
    text('stat-size', formatBytes(modelBytes));
    sizeModelFp32.textContent = `${formatBytes(modelBytes)} (${n.toLocaleString()} × 4B)`;
    sizeModelFp16.textContent = formatBytes(n * 2);
    sizeModelInt8.textContent = formatBytes(n * 1);

    const rawBytes = state.trainRes * state.trainRes * 3;
    sizeTargetRaw.textContent = `${formatBytes(rawBytes)} (${state.trainRes}×${state.trainRes})`;
    sizeTargetPng.textContent = targetPngBytes > 0 ? formatBytes(targetPngBytes) : 'measuring…';
    sizeTargetJpeg.textContent =
      targetJpegBytes > 0 ? formatBytes(targetJpegBytes) : 'measuring…';
    sizeRatioPng.textContent = targetPngBytes > 0 ? formatRatio(modelBytes, targetPngBytes) : '—';
    sizeRatioJpeg.textContent =
      targetJpegBytes > 0 ? formatRatio(modelBytes, targetJpegBytes) : '—';
  }

  function measureCompressionSizes() {
    if (compressionMeasurePending) return;
    compressionMeasurePending = true;
    void (async () => {
      // Encodes the same downsampled canvas the trainer learns from, so the
      // comparison is apples-to-apples with what the model is actually
      // fitting. Sequential rather than concurrent: both calls downsample
      // onto one shared scratch canvas, and overlapping toBlob() encodes
      // against it could race.
      targetPngBytes = await paint.toPngBytes(state.trainRes);
      targetJpegBytes = await paint.toJpegBytes(state.trainRes, JPEG_QUALITY);
      compressionMeasurePending = false;
      updateSizeStats();
    })();
  }

  const zoomCtl = slider({
    label: 'View zoom',
    min: 0.2,
    max: 8,
    step: 0.01,
    value: 1,
    log: true,
    format: (v) => `${v.toFixed(2)}×`,
    hint: 'Above 1× you are looking outside the trained square — the field keeps going.',
    onInput: (v) => {
      state.view.zoom = v;
      trainer.setView(state.view);
    },
  });

  const freqCtl = slider({
    label: 'Frequencies',
    min: 1,
    max: 128,
    step: 1,
    value: state.spec.frequencies,
    format: (v) => String(v),
    onInput: (v) => {
      state.spec.frequencies = v;
      rebuildModel();
    },
  });
  const scaleCtl = slider({
    label: 'Encoding bandwidth',
    min: 0.25,
    max: 32,
    step: 0.25,
    value: state.spec.encodingScale,
    log: true,
    format: (v) => v.toFixed(2),
    hint: 'Higher = sharper detail, but noisier and slower to settle.',
    onInput: (v) => {
      state.spec.encodingScale = v;
      rebuildModel();
    },
  });
  const omegaCtl = slider({
    label: 'SIREN ω₀',
    min: 1,
    max: 60,
    step: 1,
    value: state.spec.omega0,
    format: (v) => String(v),
    onInput: (v) => {
      state.spec.omega0 = v;
      rebuildModel();
    },
  });

  function syncEncodingControls() {
    const enc = state.spec.encoding;
    freqCtl.setDisabled(enc === 'none');
    scaleCtl.setDisabled(enc === 'none');
    omegaCtl.setDisabled(state.spec.activation !== 'sine');
  }

  const controls = must('controls');
  controls.append(
    fieldset('Architecture', [
      el('div', { class: 'shape-row' }, [shapeLabel]),
      slider({
        label: 'Hidden layers',
        min: 1,
        max: 8,
        step: 1,
        value: state.spec.hiddenLayers,
        format: (v) => String(v),
        onInput: (v) => {
          state.spec.hiddenLayers = v;
          rebuildModel();
        },
      }).root,
      slider({
        label: 'Width',
        min: 4,
        max: 256,
        step: 4,
        value: state.spec.width,
        format: (v) => String(v),
        onInput: (v) => {
          state.spec.width = v;
          rebuildModel();
        },
      }).root,
      select<Activation>({
        label: 'Activation',
        options: ACTIVATIONS,
        value: state.spec.activation,
        labels: { sine: 'sine (SIREN)' },
        onChange: (v) => {
          state.spec.activation = v;
          syncEncodingControls();
          rebuildModel();
        },
      }),
      omegaCtl.root,
    ]),

    fieldset('Input encoding', [
      select<Encoding>({
        label: 'Encoding',
        options: ENCODINGS,
        value: state.spec.encoding,
        labels: { none: 'none (raw x, y)', positional: 'positional (octaves)', gaussian: 'gaussian random' },
        hint: 'Without an encoding an MLP can only fit smooth, blurry images.',
        onChange: (v) => {
          state.spec.encoding = v;
          syncEncodingControls();
          rebuildModel();
        },
      }),
      freqCtl.root,
      scaleCtl.root,
    ]),

    fieldset('Optimisation', [
      slider({
        label: 'Learning rate',
        min: 1e-5,
        max: 0.1,
        step: 1,
        value: state.hyper.lr,
        log: true,
        format: (v) => v.toExponential(1),
        onInput: (v) => {
          state.hyper.lr = v;
          trainer.setHyper(state.hyper);
        },
      }).root,
      slider({
        label: 'Batch size',
        min: 64,
        max: 16384,
        step: 1,
        value: state.hyper.batch,
        log: true,
        format: (v) => String(Math.round(v)),
        hint: 'Pixels sampled per step.',
        onInput: (v) => {
          state.hyper.batch = Math.max(64, Math.round(v));
          trainer.setHyper(state.hyper);
        },
      }).root,
      slider({
        label: 'Weight decay',
        min: 0,
        max: 0.01,
        step: 0.0001,
        value: state.hyper.weightDecay,
        format: (v) => v.toFixed(4),
        onInput: (v) => {
          state.hyper.weightDecay = v;
          trainer.setHyper(state.hyper);
        },
      }).root,
      slider({
        label: 'Steps / frame',
        min: 1,
        max: 64,
        step: 1,
        value: state.stepsPerFrame,
        format: (v) => String(v),
        onInput: (v) => (state.stepsPerFrame = v),
      }).root,
    ]),

    fieldset('Resolution & view', [
      select<string>({
        label: 'Target resolution',
        options: ['32', '64', '128', '256'],
        value: String(state.trainRes),
        hint: 'The painting is downsampled to this grid before training.',
        onChange: (v) => {
          state.trainRes = Number(v);
          uploadTarget(true);
        },
      }),
      select<string>({
        label: 'Field resolution',
        options: ['64', '128', '256', '384', '512'],
        value: String(state.outRes),
        onChange: (v) => {
          state.outRes = Number(v);
          trainer.setOutputSize(state.outRes, state.outRes);
        },
      }),
      zoomCtl.root,
    ]),

    fieldset('Model I/O', [
      el('div', { class: 'row wrap' }, [
        button('Export JSON', () => exportModel()),
        button('Import JSON', () => importInput.click()),
        button('New seed', () => {
          state.spec.seed = (Math.random() * 1e9) | 0;
          rebuildModel();
        }),
      ]),
    ]),

    fieldset('Size & compression', [
      kvRow('Model, fp32 (as trained)', sizeModelFp32),
      kvRow('if quantised to fp16', sizeModelFp16, 'sub'),
      kvRow('if quantised to int8', sizeModelInt8, 'sub'),
      kvSeparator(),
      kvRow('Target, raw RGB8', sizeTargetRaw),
      kvRow('Target, PNG', sizeTargetPng),
      kvRow('Target, JPEG (q=0.8)', sizeTargetJpeg),
      kvSeparator(),
      kvRow('Model vs. PNG', sizeRatioPng),
      kvRow('Model vs. JPEG', sizeRatioJpeg),
      el('p', { class: 'muted' }, [
        'fp16/int8 are hypothetical — this app trains and exports fp32 only. ' +
          'A network smaller than PNG/JPEG is, loosely, doing better than general-purpose compression for this image ' +
          '(though JPEG at q=0.8 is lossy too, so it’s not quite an apples-to-apples fidelity comparison).',
      ]),
    ]),
  );

  const importInput = el('input', { type: 'file', accept: 'application/json', class: 'hidden-file' });
  importInput.addEventListener('change', async () => {
    const f = importInput.files?.[0];
    importInput.value = '';
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as { spec: ModelSpec; params: number[] };
      const layout = buildLayout(parsed.spec);
      if (layout.paramCount !== parsed.params.length) {
        throw new Error('parameter count does not match the saved architecture');
      }
      state.spec = { ...parsed.spec };
      const old = trainer;
      trainer = makeTrainer();
      old.destroy();
      trainer.setParams(new Float32Array(parsed.params));
      trainer.setImage(lastTarget, state.trainRes, state.trainRes);
      trainer.setView(state.view);
      onLayoutChanged();
      rebuildControlsFromState();
    } catch (err) {
      alert(`Could not import model: ${(err as Error).message}`);
    }
  });
  controls.append(importInput);

  function rebuildControlsFromState() {
    freqCtl.set(state.spec.frequencies);
    scaleCtl.set(state.spec.encodingScale);
    omegaCtl.set(state.spec.omega0);
    syncEncodingControls();
  }

  function exportModel() {
    const blob = new Blob(
      [JSON.stringify({ spec: state.spec, params: Array.from(trainer.getParams()) })],
      { type: 'application/json' },
    );
    downloadBlob(blob, 'neuralite-model.json');
  }

  syncEncodingControls();

  // ------------------------------------------------------------ top bar

  const playBtn = must('btn-play') as HTMLButtonElement;
  const setRunning = (r: boolean) => {
    state.running = r;
    playBtn.textContent = r ? 'Pause' : 'Resume';
    playBtn.classList.toggle('paused', !r);
  };
  playBtn.addEventListener('click', () => setRunning(!state.running));
  (must('btn-reinit') as HTMLButtonElement).addEventListener('click', () => rebuildModel());

  window.addEventListener('keydown', (e) => {
    const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement;
    if (typing) return;
    if (e.code === 'Space') {
      e.preventDefault();
      setRunning(!state.running);
    } else if (e.key === 'z' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (e.shiftKey) paint.redo();
      else paint.undo();
    }
  });

  text('stat-backend', gpu ? 'webgpu' : 'cpu (fallback)');
  if (!gpu) must('stat-backend').classList.add('warn');

  // ------------------------------------------------------------ target upload

  let lastTarget: Float32Array = new Float32Array(0);
  let targetDirty = true;
  let lastUpload = 0;

  function uploadTarget(force = false) {
    lastTarget = paint.toTarget(state.trainRes);
    trainer.setImage(lastTarget, state.trainRes, state.trainRes);
    targetDirty = false;
    if (force) lastUpload = 0;
    updateSizeStats();
    measureCompressionSizes();
  }

  onLayoutChanged();
  uploadTarget(true);
  paint.onChange = () => (targetDirty = true);

  // ------------------------------------------------------------ loop

  const lossHistory: number[] = [];
  const lossCanvas = must('loss-chart') as HTMLCanvasElement;
  const lossCtx = lossCanvas.getContext('2d')!;

  let lastParamRefresh = 0;
  let lastNeuronRefresh = 0;
  let lastRateSample = performance.now();
  let stepsAtSample = 0;

  function frame(now: number) {
    if (targetDirty && now - lastUpload > TARGET_UPLOAD_MS) {
      uploadTarget();
      lastUpload = now;
    }

    trainer.tick(state.running ? state.stepsPerFrame : 0, true);

    const mse = trainer.loss();
    if (state.running && mse > 0) {
      lossHistory.push(mse);
      if (lossHistory.length > 480) lossHistory.shift();
    }

    text('stat-steps', trainer.stepCount.toLocaleString());
    text('stat-loss', mse > 0 ? mse.toExponential(2) : '—');
    text('stat-psnr', mse > 0 ? `${(10 * Math.log10(1 / mse)).toFixed(1)} dB` : '—');

    if (now - lastRateSample > 500) {
      const rate = ((trainer.stepCount - stepsAtSample) / (now - lastRateSample)) * 1000;
      text('stat-rate', rate.toFixed(0));
      lastRateSample = now;
      stepsAtSample = trainer.stepCount;
    }

    if (now - lastParamRefresh > PARAM_REFRESH_MS) {
      lastParamRefresh = now;
      void trainer.refreshParams().then((p) => {
        arch.update(p);
        weights.update(p);
      });
    }

    if (inspectTab === 'neurons' && now - lastNeuronRefresh > NEURON_REFRESH_MS) {
      lastNeuronRefresh = now;
      void trainer.neuronPreview(neurons.activeLayer, PREVIEW_GRID).then((p) => {
        if (p) neurons.update(p);
      });
    }

    drawLoss(lossCtx, lossCanvas, lossHistory);
    requestAnimationFrame(frame);
  }

  setRunning(true);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- helpers

function must(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node;
}

function text(id: string, value: string) {
  must(id).textContent = value;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function drawLoss(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, history: number[]) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  // Log scale: loss spans orders of magnitude over a run.
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of history) {
    const l = Math.log10(Math.max(v, 1e-8));
    if (l < lo) lo = l;
    if (l > hi) hi = l;
  }
  if (hi - lo < 0.3) {
    const mid = (hi + lo) / 2;
    lo = mid - 0.15;
    hi = mid + 0.15;
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = (h * i) / 3 + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = '#5eead4';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * w;
    const y = h - ((Math.log10(Math.max(v, 1e-8)) - lo) / (hi - lo)) * (h - 6) - 3;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.font = '10px ui-monospace, monospace';
  ctx.fillText(`mse ${history[history.length - 1].toExponential(2)}`, 6, 12);
}

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadCanvas(canvas: HTMLCanvasElement) {
  canvas.toBlob((blob) => {
    if (blob) downloadBlob(blob, 'neuralite-field.png');
  });
}

boot().catch((err) => {
  const fatal = document.getElementById('fatal');
  if (fatal) {
    fatal.hidden = false;
    fatal.textContent = `neuralite failed to start: ${(err as Error).message}`;
  }
  throw err;
});
