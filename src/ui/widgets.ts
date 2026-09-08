export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

export function svgEl(tag: string, attrs: Record<string, string | number> = {}): SVGElement {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

interface SliderSpec {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  /** Map slider position to value logarithmically (for learning rate etc.). */
  log?: boolean;
  format?: (v: number) => string;
  hint?: string;
  onInput: (v: number) => void;
}

export interface Control {
  root: HTMLElement;
  set(v: number): void;
  setDisabled(disabled: boolean): void;
}

export function slider(spec: SliderSpec): Control {
  const fmt = spec.format ?? ((v: number) => String(v));
  const toPos = (v: number) =>
    spec.log
      ? ((Math.log(v) - Math.log(spec.min)) / (Math.log(spec.max) - Math.log(spec.min))) * 1000
      : v;
  const toVal = (p: number) => {
    if (!spec.log) return p;
    const v = Math.exp(
      Math.log(spec.min) + (p / 1000) * (Math.log(spec.max) - Math.log(spec.min)),
    );
    // Snap to two significant figures so the readout stays legible.
    const mag = Math.pow(10, Math.floor(Math.log10(v)) - 1);
    return Math.round(v / mag) * mag;
  };

  const value = el('span', { class: 'ctl-value' }, [fmt(spec.value)]);
  const input = el('input', {
    type: 'range',
    min: String(spec.log ? 0 : spec.min),
    max: String(spec.log ? 1000 : spec.max),
    step: String(spec.log ? 1 : spec.step),
    value: String(toPos(spec.value)),
  });

  input.addEventListener('input', () => {
    const v = toVal(Number(input.value));
    value.textContent = fmt(v);
    spec.onInput(v);
  });

  const head = el('div', { class: 'ctl-head' }, [
    el('label', {}, [spec.label]),
    value,
  ]);
  const root = el('div', { class: 'ctl' }, [head, input]);
  if (spec.hint) root.append(el('p', { class: 'ctl-hint' }, [spec.hint]));

  return {
    root,
    set(v: number) {
      input.value = String(toPos(v));
      value.textContent = fmt(v);
    },
    setDisabled(disabled: boolean) {
      input.disabled = disabled;
      root.classList.toggle('disabled', disabled);
    },
  };
}

interface SelectSpec<T extends string> {
  label: string;
  options: readonly T[];
  value: T;
  labels?: Partial<Record<T, string>>;
  hint?: string;
  onChange: (v: T) => void;
}

export function select<T extends string>(spec: SelectSpec<T>): HTMLElement {
  const sel = el('select', {});
  for (const o of spec.options) {
    const opt = el('option', { value: o }, [spec.labels?.[o] ?? o]);
    sel.append(opt);
  }
  sel.value = spec.value;
  sel.addEventListener('change', () => spec.onChange(sel.value as T));

  const root = el('div', { class: 'ctl' }, [
    el('div', { class: 'ctl-head' }, [el('label', {}, [spec.label])]),
    sel,
  ]);
  if (spec.hint) root.append(el('p', { class: 'ctl-hint' }, [spec.hint]));
  return root;
}

export function segmented<T extends string>(
  options: readonly { value: T; label: string; title?: string }[],
  value: T,
  onChange: (v: T) => void,
): HTMLElement {
  const root = el('div', { class: 'segmented' });
  const buttons = new Map<T, HTMLButtonElement>();
  for (const o of options) {
    const b = el('button', o.title ? { title: o.title } : {}, [o.label]);
    b.addEventListener('click', () => {
      for (const [, other] of buttons) other.classList.remove('on');
      b.classList.add('on');
      onChange(o.value);
    });
    if (o.value === value) b.classList.add('on');
    buttons.set(o.value, b);
    root.append(b);
  }
  return root;
}

export function button(
  label: string,
  onClick: () => void,
  cls = '',
): HTMLButtonElement {
  const b = el('button', cls ? { class: cls } : {}, [label]);
  b.addEventListener('click', onClick);
  return b;
}

export function fieldset(title: string, children: (Node | string)[]): HTMLElement {
  return el('section', { class: 'group' }, [
    el('h3', { class: 'group-title' }, [title]),
    ...children,
  ]);
}

/** A label/value line, e.g. for a stats readout. Pass the value element in to keep a live reference to it. */
export function kvRow(label: string, valueNode: Node, cls = ''): HTMLElement {
  return el('div', { class: cls ? `kv-row ${cls}` : 'kv-row' }, [
    el('span', { class: 'kv-label' }, [label]),
    valueNode,
  ]);
}

export function kvSeparator(): HTMLElement {
  return el('div', { class: 'kv-sep' });
}
