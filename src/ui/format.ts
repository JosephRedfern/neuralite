/** Human-readable byte count, e.g. "812 B", "49.5 KB", "1.2 MB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ['KB', 'MB', 'GB'];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${units[i]}`;
}

/** "5.7× larger" / "3.2× smaller", comparing `a` against reference `b`. */
export function formatRatio(a: number, b: number): string {
  if (b <= 0 || a <= 0) return '—';
  const r = a / b;
  return r >= 1 ? `${r.toFixed(1)}× larger` : `${(1 / r).toFixed(1)}× smaller`;
}
