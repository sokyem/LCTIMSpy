import Plotly from "plotly.js-cartesian-dist-min";
import type { Config, Data, Layout, PlotlyHTMLElement } from "plotly.js";
import { useEffect, useRef } from "react";

import { cssVar } from "../lib/theme";

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  /** Changes when the theme does, so chrome colours are re-read. */
  dark: boolean;
  ariaLabel: string;
  onSelected?: (event: unknown) => void;
  onRelayout?: (payload: Record<string, unknown>) => void;
}

/** Plotly chrome built from the design tokens: recessive grid and axes, text in text ink. */
export function baseLayout(): Partial<Layout> {
  const text = cssVar("--text");
  const muted = cssVar("--text-muted");
  const grid = cssVar("--grid");
  const axis = cssVar("--axis");
  const axisStyle = {
    gridcolor: grid,
    linecolor: axis,
    zerolinecolor: axis,
    tickcolor: axis,
    tickfont: { color: muted, size: 11 },
    title: { font: { color: muted, size: 12 } },
    showline: true,
    ticks: "outside" as const,
    ticklen: 4,
  };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: cssVar("--font"), color: text, size: 12 },
    margin: { l: 64, r: 16, t: 12, b: 44 },
    xaxis: { ...axisStyle },
    yaxis: { ...axisStyle, exponentformat: "SI" },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: text, size: 12 } },
    hoverlabel: {
      bgcolor: cssVar("--surface"),
      bordercolor: cssVar("--border-strong"),
      font: { color: text, family: cssVar("--font"), size: 12 },
    },
  };
}

function merge(base: Partial<Layout>, over: Partial<Layout> = {}): Partial<Layout> {
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(over)) {
    const b = (base as Record<string, unknown>)[k];
    out[k] = b && v && typeof b === "object" && typeof v === "object" && !Array.isArray(v) ? { ...b, ...v } : v;
  }
  return out as Partial<Layout>;
}

export function Plot({ data, layout, config, className, dark, ariaLabel, onSelected, onRelayout }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const handlers = useRef({ onSelected, onRelayout });
  handlers.current = { onSelected, onRelayout };

  const attached = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    void Plotly.react(el, data, merge(baseLayout(), layout), {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["lasso2d", "toImage"],
      ...config,
    }).then((gd: PlotlyHTMLElement) => {
      // Plotly adds .on() to the element on first plot; attach once, through a
      // ref so handlers always see the latest props.
      if (attached.current) return;
      attached.current = true;
      gd.on("plotly_selected", (e) => handlers.current.onSelected?.(e));
      gd.on("plotly_relayout", (e) => handlers.current.onRelayout?.(e as unknown as Record<string, unknown>));
    });
    // `dark` is a dependency so chrome colours are re-read from the CSS tokens.
  }, [data, layout, config, dark]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      if ((el as unknown as { _fullLayout?: unknown })._fullLayout) void Plotly.Plots.resize(el);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      Plotly.purge(el); // also drops the event listeners, so let the next plot re-attach
      attached.current = false;
    };
  }, []);

  return <div ref={ref} className={`plot ${className ?? ""}`} role="img" aria-label={ariaLabel} />;
}
