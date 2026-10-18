// Categorical palette (validated: scripts/validate_palette.js, light on #ffffff and
// dark on #111a2e). The ORDER is the colour-vision-deficiency safety mechanism, so
// slots are assigned in this order and never cycled: there is no 9th colour.
export const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
export const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const MAX_SERIES = SERIES_LIGHT.length;

// RT windows are identified by label and line dash, never by colour, so they
// never collide with target colours. Colour always means "which target".
export const WINDOW_DASH = ["solid", "dash", "dot", "dashdot"] as const;
export const MAX_WINDOWS = WINDOW_DASH.length;

export function seriesColor(slot: number | null | undefined, dark: boolean): string | null {
  if (slot == null || slot < 0 || slot >= MAX_SERIES) return null;
  return (dark ? SERIES_DARK : SERIES_LIGHT)[slot];
}

/** First slot not used by any of ``taken``, or null when all 8 are in use. */
export function freeSlot(taken: (number | null | undefined)[]): number | null {
  for (let i = 0; i < MAX_SERIES; i++) if (!taken.includes(i)) return i;
  return null;
}

export const windowName = (i: number) => `W${i + 1}`;
