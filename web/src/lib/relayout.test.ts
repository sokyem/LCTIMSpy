import { describe, expect, it } from "vitest";

import { breakGaps, normalize, rangeFromSelection, windowEditsFromRelayout } from "./relayout";

const W = [
  { start: 18.3, end: 19.0 },
  { start: 21.5, end: 22.2 },
];

describe("windowEditsFromRelayout", () => {
  it("reads per-key shape edits (drag one edge)", () => {
    const edits = windowEditsFromRelayout({ "shapes[1].x1": 22.55 }, W);
    expect([...edits]).toEqual([[1, { start: 21.5, end: 22.55 }]]);
  });

  it("reads both edges of a moved band", () => {
    const edits = windowEditsFromRelayout({ "shapes[0].x0": 18.4, "shapes[0].x1": 19.1 }, W);
    expect(edits.get(0)).toEqual({ start: 18.4, end: 19.1 });
  });

  it("reads the whole-array payload some Plotly versions send", () => {
    const edits = windowEditsFromRelayout({ shapes: [{ x0: 18.0, x1: 18.9 }, { x0: "21.6", x1: "22.4" }] }, W);
    expect(edits.get(0)).toEqual({ start: 18.0, end: 18.9 });
    expect(edits.get(1)).toEqual({ start: 21.6, end: 22.4 });
  });

  it("re-orders an inside-out drag and drops zero-width ones", () => {
    expect(windowEditsFromRelayout({ "shapes[0].x0": 19.5 }, W).get(0)).toEqual({ start: 19.0, end: 19.5 });
    expect(windowEditsFromRelayout({ "shapes[0].x0": 19.0 }, W).size).toBe(0);
  });

  it("ignores zoom, pan, and shapes we do not own", () => {
    expect(windowEditsFromRelayout({ "xaxis.range[0]": 10, "xaxis.range[1]": 30 }, W).size).toBe(0);
    expect(windowEditsFromRelayout({ "shapes[5].x0": 1 }, W).size).toBe(0);
    expect(windowEditsFromRelayout({ "xaxis.autorange": true }, W).size).toBe(0);
  });

  it("rounds to 0.001 min", () => {
    expect(windowEditsFromRelayout({ "shapes[0].x1": 19.123456 }, W).get(0)?.end).toBe(19.123);
  });
});

describe("rangeFromSelection", () => {
  it("returns the ordered x range of a box selection", () => {
    expect(rangeFromSelection({ range: { x: [19.2, 18.4] } })).toEqual({ start: 18.4, end: 19.2 });
  });
  it("returns null for clicks, lasso, or empty events", () => {
    expect(rangeFromSelection(undefined)).toBeNull();
    expect(rangeFromSelection({ lassoPoints: {} })).toBeNull();
    expect(rangeFromSelection({ range: { x: [5, 5] } })).toBeNull();
  });
});

describe("normalize", () => {
  it("matches the Python core: min-max, flat series to 0", () => {
    expect(normalize([2, 4, 6])).toEqual([0, 0.5, 1]);
    expect(normalize([3, 3])).toEqual([0, 0]);
    expect(normalize([])).toEqual([]);
  });
});

describe("breakGaps", () => {
  it("breaks the line across empty mobility bins", () => {
    const x = [0.9, 0.9005, 0.901, 0.9015, 1.2];
    const y = [1, 2, 3, 2, 1];
    expect(breakGaps(x, y)).toEqual({ x: [0.9, 0.9005, 0.901, 0.9015, null, 1.2], y: [1, 2, 3, 2, null, 1] });
  });
  it("leaves evenly spaced and short series alone", () => {
    expect(breakGaps([1, 2, 3, 4], [1, 1, 1, 1]).x).toEqual([1, 2, 3, 4]);
    expect(breakGaps([1, 9], [1, 1]).x).toEqual([1, 9]);
  });
});
