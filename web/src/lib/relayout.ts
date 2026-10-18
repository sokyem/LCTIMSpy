// Parsing Plotly events for the RT-window interaction. Kept pure so it is unit-tested.

export interface RtWindow {
  start: number;
  end: number;
}

/**
 * Window edits from a ``plotly_relayout`` payload.
 *
 * Dragging or resizing an editable shape reports either per-key updates
 * (``{"shapes[1].x0": 18.2, "shapes[1].x1": 19.0}``) or, in some Plotly versions
 * and for some gestures, the whole ``shapes`` array. Both are handled; anything
 * else (zoom, pan, autoscale) yields no edits. Bounds are re-ordered so a window
 * dragged "inside out" stays valid.
 */
export function windowEditsFromRelayout(
  payload: Record<string, unknown>,
  current: RtWindow[],
): Map<number, RtWindow> {
  const edits = new Map<number, RtWindow>();
  const get = (i: number) => edits.get(i) ?? { ...current[i] };

  const whole = payload["shapes"];
  if (Array.isArray(whole)) {
    whole.forEach((shape, i) => {
      if (i >= current.length || !shape || typeof shape !== "object") return;
      const s = shape as { x0?: unknown; x1?: unknown };
      const x0 = toNumber(s.x0);
      const x1 = toNumber(s.x1);
      if (x0 == null || x1 == null) return;
      edits.set(i, { start: x0, end: x1 });
    });
  }

  for (const [key, value] of Object.entries(payload)) {
    const m = /^shapes\[(\d+)\]\.(x0|x1)$/.exec(key);
    if (!m) continue;
    const i = Number(m[1]);
    const v = toNumber(value);
    if (i >= current.length || v == null) continue;
    const w = get(i);
    if (m[2] === "x0") w.start = v;
    else w.end = v;
    edits.set(i, w);
  }

  for (const [i, w] of edits) {
    const start = Math.min(w.start, w.end);
    const end = Math.max(w.start, w.end);
    if (end - start <= 0) edits.delete(i);
    else edits.set(i, { start: round(start), end: round(end) });
  }
  return edits;
}

/** The horizontal range of a ``plotly_selected`` box selection, if any. */
export function rangeFromSelection(event: unknown): RtWindow | null {
  if (!event || typeof event !== "object") return null;
  const range = (event as { range?: { x?: unknown } }).range;
  const x = range?.x;
  if (!Array.isArray(x) || x.length !== 2) return null;
  const a = toNumber(x[0]);
  const b = toNumber(x[1]);
  if (a == null || b == null || a === b) return null;
  return { start: round(Math.min(a, b)), end: round(Math.max(a, b)) };
}

function toNumber(v: unknown): number | null {
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
}

/** RT in minutes to 0.001 min (60 ms): finer than any LC peak, and readable. */
export const round = (x: number) => Math.round(x * 1000) / 1000;

/** Min-max normalise to [0, 1]; a flat series maps to 0 (same rule as the Python core). */
export function normalize(y: number[]): number[] {
  if (y.length === 0) return y;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of y) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  return span === 0 ? y.map(() => 0) : y.map((v) => (v - lo) / span);
}

/**
 * Insert a ``null`` wherever consecutive x values are further apart than
 * ``factor`` times the typical (median) spacing, so Plotly breaks the line
 * instead of drawing a straight segment across bins that have no signal.
 */
export function breakGaps(x: number[], y: number[], factor = 3): { x: (number | null)[]; y: (number | null)[] } {
  if (x.length < 3) return { x, y };
  const steps = x.slice(1).map((v, i) => v - x[i]).filter((d) => d > 0).sort((a, b) => a - b);
  const median = steps[Math.floor(steps.length / 2)];
  if (!median) return { x, y };
  const ox: (number | null)[] = [x[0]];
  const oy: (number | null)[] = [y[0]];
  for (let i = 1; i < x.length; i++) {
    if (x[i] - x[i - 1] > factor * median) {
      ox.push(null);
      oy.push(null);
    }
    ox.push(x[i]);
    oy.push(y[i]);
  }
  return { x: ox, y: oy };
}
