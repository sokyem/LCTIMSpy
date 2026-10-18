import { beforeEach, describe, expect, it } from "vitest";

import { MAX_SERIES } from "./lib/palette";
import { useStore } from "./store";

const s = () => useStore.getState();

beforeEach(() => {
  s().clearTargets();
  s().setWindows([]);
  s().arm(0);
});

describe("target colour slots", () => {
  it("keeps survivors' colours when another target is disabled", () => {
    s().addTargets([{ mz: 100, label: "a" }, { mz: 200, label: "b" }, { mz: 300, label: "c" }]);
    expect(s().targets.map((t) => t.slot)).toEqual([0, 1, 2]);
    s().setTargetEnabled(0, false);
    expect(s().targets.map((t) => t.slot)).toEqual([null, 1, 2]);
    // Re-enabling takes the first free slot, never repaints the others.
    s().addTargets([{ mz: 400, label: "d" }]);
    expect(s().targets.map((t) => t.slot)).toEqual([null, 1, 2, 0]);
  });

  it("never assigns a ninth colour", () => {
    const many = Array.from({ length: MAX_SERIES + 2 }, (_, i) => ({ mz: 100 + i, label: null }));
    const warning = s().addTargets(many);
    expect(warning).toMatch(/not plotted/);
    expect(s().targets.filter((t) => t.enabled)).toHaveLength(MAX_SERIES);
    expect(s().targets.at(-1)).toMatchObject({ enabled: false, slot: null });
    expect(s().setTargetEnabled(MAX_SERIES, true)).toMatch(/At most 8/);
  });

  it("skips duplicate m/z values", () => {
    s().addTargets([{ mz: 500.25, label: "x" }, { mz: 500.25, label: "dup" }]);
    expect(s().targets).toHaveLength(1);
  });
});

describe("RT window arming", () => {
  it("fills W1 then auto-arms W2, then stops arming", () => {
    s().fillArmed({ start: 1, end: 2 });
    expect(s().armed).toBe(1);
    s().fillArmed({ start: 3, end: 4 });
    expect(s().windows).toEqual([{ start: 1, end: 2 }, { start: 3, end: 4 }]);
    expect(s().armed).toBeNull();
  });

  it("redraws an armed existing window in place", () => {
    s().setWindows([{ start: 1, end: 2 }, { start: 3, end: 4 }]);
    s().arm(0);
    s().fillArmed({ start: 1.5, end: 2.5 });
    expect(s().windows[0]).toEqual({ start: 1.5, end: 2.5 });
    expect(s().windows).toHaveLength(2);
  });
});
