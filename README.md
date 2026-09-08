
# neuralite

**Note: This whole application is entirely vibe-coded - I was mostly interested in the outcome and exploring the
latent space than I was in the implementation, which is pretty much disposable.**

A browser playground for **neural fields**: paint on the left, and an MLP
`g(x, y) → rgb` learns to reproduce it on the right, live. The model's
architecture, weights and per-neuron activations are all visible — and editable —
while it trains.

```
npm install
npm run dev
```

Live at **https://josephredfern.github.io/neuralite/**, rebuilt automatically
by `.github/workflows/deploy.yml` on every push to `main` (typecheck + build +
gradient/fit checks, then deploy to Pages — see [Verifying the maths](#verifying-the-maths)).

## What it actually does

Every training step samples a batch of random pixels from the target, encodes
their coordinates, runs them through an MLP, and takes an Adam step on the mean
squared error. There is no autodiff library — forward, backward and the
optimiser are written out explicitly, once in WGSL compute shaders and once in
plain TypeScript.

- **WebGPU** (`src/model/gpu.ts`) is the fast path. One thread per output
  element, per-layer constants supplied through a dynamic-offset uniform, so
  changing the architecture never recompiles a shader.
- **CPU** (`src/model/cpu.ts`) is the fallback and the reference implementation.
  It runs the identical maths in `Float32Array`s.

The backend in use is shown in the header.

## Controls worth playing with

| Control | Why it matters |
| --- | --- |
| **Encoding** | The single biggest lever. With `none`, an MLP on raw `(x, y)` can only produce smooth blurs — this is spectral bias, and it is very visible. `gaussian` (random Fourier features) or `positional` (octaves) fixes it. |
| **Encoding bandwidth** | Low = blurry, high = sharp but noisy and slow to settle. |
| **Activation** | `sine` is SIREN: it fits high frequencies with *no* encoding at all. Try `sine` + `none`. |
| **View zoom** | Above 1× you are looking outside the square the model was trained on. What the field does out there tells you a lot about what it actually learned. |
| **Hidden layers / width** | Changing either reinitialises the model. |

Space bar pauses. ⌘Z / ⌘⇧Z undo and redo on the paint canvas.

## Inspecting and tweaking the model

- **Architecture** — columns of neurons, edges coloured by weight sign and
  magnitude. Hover an edge for its value; click a neuron to focus it. Large
  layers are drawn truncated.
- **Weights** — a heatmap of the selected layer's matrix, with the bias as a
  detached column on the right. Click any cell to edit it with a slider, or use
  the layer-wide operations (zero a unit, rescale, prune small weights, add
  noise). Edits go straight into the live model; pause first if you want them to
  stick, since Adam will otherwise pull them back.
- **Neurons** — each unit's activation over the domain. This is the view that
  makes a neural field legible: the first hidden layer shows the Fourier basis,
  later layers show what got composed out of it.

Export/Import JSON round-trips the architecture plus the weights.

## Verifying the maths

```
npm run check
```

Finite-difference checks the CPU trainer's gradients for every
activation × encoding combination, and asserts that each configuration actually
fits a synthetic target.

The check differences along whole directions (the gradient direction plus random
ones) rather than one parameter at a time, and scales the error by ‖g‖. Both
details matter: parameters are float32 and the loss is averaged over
`batch × channels`, so a single component's effect on the loss sits near the
round-off floor, and a random direction can land nearly orthogonal to the
gradient and make a correct derivative look wrong.

Current worst-case agreement is ~7e-4 ‖g‖.

Note that this exercises the CPU path. The WGSL kernels mirror it line for line
but are only validated by running the app — the training bind group originally
used 11 storage buffers, one over the 8 guaranteed by the WebGPU spec, which
silently invalidated the whole compute pass. Fixed by fusing `weightGrad`
into the Adam step and packing `mom1`/`mom2` and `batchIdx`/loss-partials into
shared buffers, bringing it down to 8.

## Layout

```
src/model/spec.ts     model definition, parameter layout, init, encodings
src/model/trainer.ts  the interface both backends implement
src/model/wgsl.ts     WGSL for training, field rendering, and the blit
src/model/gpu.ts      WebGPU backend
src/model/cpu.ts      reference backend
src/paint.ts          target canvas, brushes, image upload
src/viz/              architecture / weights / neuron views
src/main.ts           wiring, controls, the frame loop
```
