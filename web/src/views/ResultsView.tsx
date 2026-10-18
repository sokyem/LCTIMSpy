import type { Data, Layout, Shape, Annotations } from "plotly.js";
import { BarChart3, CheckCircle2, Download, Loader2, MinusCircle } from "lucide-react";
import { useMemo } from "react";

import { api, download, type Compare, type DiffRow, type MobilityDiff, type PairRow } from "../api/client";
import { Plot } from "../components/Plot";
import { Banner, Empty } from "../components/ui";
import { fmtMobility, fmtMz, fmtP, stars } from "../lib/format";
import { seriesColor, windowName } from "../lib/palette";
import { useDarkMode } from "../lib/theme";
import { useFetch } from "../lib/useFetch";
import { enabledTargets, useReadySamples, useStore } from "../store";

const METRIC_LABEL = {
  mobility_difference: "Δ mean mobility (1/K0)",
  peak_mobility_difference: "Δ peak mobility (1/K0)",
} as const;

export function ResultsView() {
  const dark = useDarkMode();
  const samples = useReadySamples();
  const targets = useStore((s) => s.targets);
  const windows = useStore((s) => s.windows);
  const offsets = useStore((s) => s.offsets);
  const query = useStore((s) => s.query);
  const stats = useStore((s) => s.stats);
  const notify = useStore((s) => s.notify);
  const setView = useStore((s) => s.setView);

  const tgs = useMemo(() => enabledTargets(targets), [targets]);
  const withStats = samples.length >= stats.min_samples;
  const req =
    samples.length && tgs.length && windows.length >= 2
      ? {
          dataset_ids: samples.map((d) => d.id),
          targets: tgs.map((t) => ({ mz: t.mz, label: t.label })),
          windows,
          query,
          rt_offsets: Object.fromEntries(samples.map((d) => [d.id, offsets[d.id] ?? 0])),
          ...(withStats ? { stats, include_figure: false } : {}),
        }
      : null;

  const res = useFetch<Compare | MobilityDiff>(
    req && JSON.stringify(req),
    async (signal) => {
      if (!withStats) return api.post<MobilityDiff>("/analysis/mobility-difference", req, signal);
      try {
        return await api.post<Compare>("/analysis/compare", req, signal);
      } catch (e) {
        // Too few targets measured in enough samples: show differences without stats.
        if (e instanceof Error && /at least/.test(e.message)) {
          const { stats: _s, include_figure: _f, ...plain } = req as Record<string, unknown>;
          return api.post<MobilityDiff>("/analysis/mobility-difference", plain, signal);
        }
        throw e;
      }
    },
    250,
  );

  const compare = res.data && "pairs" in res.data ? res.data : null;
  const diff: MobilityDiff | null = compare ? compare.differences : (res.data as MobilityDiff | null);
  const slotOf = (mz: number) => tgs.find((t) => Math.abs(t.mz - mz) < 1e-9)?.slot ?? null;
  const labelOf = (mz: number) => tgs.find((t) => Math.abs(t.mz - mz) < 1e-9)?.label || `m/z ${fmtMz(mz)}`;

  // slotOf/labelOf derive from tgs, which is in the deps.
  const box = useMemo(() => buildBox(compare, stats.metric, dark, slotOf, labelOf), [compare, stats.metric, dark, tgs]);

  if (!req) {
    const missing = !samples.length ? "datasets" : !tgs.length ? "targets" : "windows";
    return (
      <div className="view">
        <h1>Results</h1>
        <div className="card">
          <Empty icon={<BarChart3 size={32} aria-hidden />} title="Not ready yet">
            <p>
              {missing === "datasets" && "Select extracted datasets first."}
              {missing === "targets" && "Choose at least one target to plot."}
              {missing === "windows" && "Set at least two RT windows."}
            </p>
            <button className="btn btn-primary" style={{ marginTop: 8 }} onClick={() => setView(missing === "windows" ? "windows" : missing)}>
              Go to {missing === "windows" ? "RT windows" : missing[0].toUpperCase() + missing.slice(1)}
            </button>
          </Empty>
        </div>
      </div>
    );
  }

  const exportDiff = () => download("mobility-difference", "csv", req).catch((e) => notify("err", String(e)));
  const exportFig = () => download("compare", "html", req).catch((e) => notify("err", String(e)));
  const exportPairs = () => download("compare", "csv", req).catch((e) => notify("err", String(e)));
  const nSig = compare?.pairs.filter((p) => p.reject).length ?? 0;

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Results</h1>
          <p className="muted">
            Mobility difference between {windows.map((_, i) => windowName(i)).join(", ")} for {samples.length} sample
            {samples.length > 1 ? "s" : ""}.{" "}
            {!withStats && `Statistics need at least ${stats.min_samples} samples.`}
          </p>
        </div>
        <div className="row">
          {res.loading && <Loader2 size={16} className="spin faint" aria-label="Computing" />}
          <button className="btn btn-sm" onClick={exportDiff}>
            <Download size={13} aria-hidden /> Differences CSV
          </button>
          {compare && (
            <>
              <button className="btn btn-sm" onClick={exportPairs}>
                <Download size={13} aria-hidden /> Stats CSV
              </button>
              <button className="btn btn-sm" onClick={exportFig}>
                <Download size={13} aria-hidden /> Figure HTML
              </button>
            </>
          )}
        </div>
      </div>

      {res.error && <Banner kind="err" title="Could not compute results">{res.error.message}</Banner>}

      {diff && (
        <div className="tiles">
          <Tile label="Samples" value={samples.length} />
          <Tile label="Targets" value={tgs.length} />
          <Tile label="Differences" value={diff.differences.length} sub={`${windows.length} windows`} />
          <Tile label="Excluded" value={diff.excluded.length} sub={diff.excluded.length ? "see reasons below" : "none"} />
          {compare && <Tile label="Significant pairs" value={`${nSig} / ${compare.pairs.length}`} sub={`${stats.correction}, α = ${stats.alpha}`} />}
        </div>
      )}

      {compare && (
        <div className="grid-side">
          <section className="card" aria-labelledby="box-h">
            <div className="card-head">
              <h2 id="box-h">{METRIC_LABEL[stats.metric]} by target</h2>
              <span className="faint" style={{ fontSize: 12 }}>
                {windowName(0)} vs {windowName(1)} · one point per sample
              </span>
            </div>
            <div className="card-body">
              <Plot className="plot-box" dark={dark} data={box.data} layout={box.layout} ariaLabel="Box plot of mobility differences by target" />
            </div>
          </section>
          <section className="card" aria-labelledby="pairs-h">
            <div className="card-head">
              <h2 id="pairs-h">Pairwise tests</h2>
            </div>
            <PairsTable pairs={compare.pairs} labelOf={labelOf} />
            <p className="faint" style={{ fontSize: 12, padding: "8px 16px 12px" }}>
              {stats.equal_var ? "Student's" : "Welch's"} t-test, {stats.correction} correction.
            </p>
          </section>
        </div>
      )}

      {diff && (
        <section className="card" aria-labelledby="diff-h">
          <div className="card-head">
            <h2 id="diff-h">Per-sample mobility</h2>
          </div>
          <DiffTable rows={diff.differences} labelOf={labelOf} slotOf={slotOf} dark={dark} />
        </section>
      )}

      {diff && diff.excluded.length > 0 && (
        <section className="card" aria-labelledby="ex-h">
          <div className="card-head">
            <h2 id="ex-h">Excluded</h2>
            <span className="faint" style={{ fontSize: 12 }}>Adjust the threshold or windows in Settings / RT windows</span>
          </div>
          <div className="table-wrap" style={{ maxHeight: 260 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>Target</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {diff.excluded.map((r, i) => (
                  <tr key={i}>
                    <td>{r.sample_name}</td>
                    <td>{labelOf(r.target_mz)}</td>
                    <td className="muted">{r.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function Tile({ label, value, sub }: { label: string; value: number | string; sub?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

function PairsTable({ pairs, labelOf }: { pairs: PairRow[]; labelOf: (mz: number) => string }) {
  if (!pairs.length) return <p className="muted" style={{ padding: 16 }}>Only one target has enough samples; nothing to compare.</p>;
  return (
    <div className="table-wrap" style={{ maxHeight: 360 }}>
      <table className="data">
        <thead>
          <tr>
            <th>Pair</th>
            <th className="r">p (adj.)</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map((p, i) => (
            <tr key={i}>
              <td>
                {labelOf(p.group_a)} <span className="faint">vs</span> {labelOf(p.group_b)}
              </td>
              <td className="r num" title={`raw p = ${fmtP(p.raw_p_value)}`}>
                {fmtP(p.adjusted_p_value)}
              </td>
              <td>
                {p.reject ? (
                  <span className="row" style={{ gap: 4, flexWrap: "nowrap", whiteSpace: "nowrap", color: "var(--success)" }}>
                    <CheckCircle2 size={14} aria-hidden /> Significant {stars(p.adjusted_p_value)}
                  </span>
                ) : (
                  <span className="row faint" style={{ gap: 4, flexWrap: "nowrap" }}>
                    <MinusCircle size={14} aria-hidden /> Not significant
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiffTable({ rows, labelOf, slotOf, dark }: {
  rows: DiffRow[];
  labelOf: (mz: number) => string;
  slotOf: (mz: number) => number | null;
  dark: boolean;
}) {
  if (!rows.length) return <p className="muted" style={{ padding: 16 }}>No target passed the checks in every window.</p>;
  const multi = rows.some((r) => r.window_a !== 0 || r.window_b !== 1);
  return (
    <div className="table-wrap" style={{ maxHeight: 420 }}>
      <table className="data">
        <thead>
          <tr>
            <th>Sample</th>
            <th>Target</th>
            {multi && <th>Windows</th>}
            <th className="r">Mean 1/K0 (A)</th>
            <th className="r">Mean 1/K0 (B)</th>
            <th className="r">Δ mean</th>
            <th className="r">Peak 1/K0 (A)</th>
            <th className="r">Peak 1/K0 (B)</th>
            <th className="r">Δ peak</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const c = seriesColor(slotOf(r.target_mz), dark);
            return (
              <tr key={i}>
                <td>{r.sample_name}</td>
                <td>
                  <span className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                    <span className={`swatch ${c ? "" : "swatch-empty"}`} style={c ? { background: c } : undefined} aria-hidden />
                    {labelOf(r.target_mz)}
                  </span>
                </td>
                {multi && <td>{windowName(r.window_a)} → {windowName(r.window_b)}</td>}
                <td className="r num">{fmtMobility(r.avg_mobility_a)}</td>
                <td className="r num">{fmtMobility(r.avg_mobility_b)}</td>
                <td className="r num"><strong>{fmtMobility(r.mobility_difference)}</strong></td>
                <td className="r num">{fmtMobility(r.peak_mobility_a)}</td>
                <td className="r num">{fmtMobility(r.peak_mobility_b)}</td>
                <td className="r num">{fmtMobility(r.peak_mobility_difference)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function buildBox(
  compare: Compare | null,
  metric: keyof typeof METRIC_LABEL,
  dark: boolean,
  slotOf: (mz: number) => number | null,
  labelOf: (mz: number) => string,
): { data: Data[]; layout: Partial<Layout> } {
  if (!compare) return { data: [], layout: {} };
  const rows = compare.data;
  const order = [...new Set(rows.map((r) => r.target_mz))];
  const data: Data[] = order.map((mz) => {
    const sub = rows.filter((r) => r.target_mz === mz);
    const color = seriesColor(slotOf(mz), dark) ?? undefined;
    return {
      type: "box",
      name: labelOf(mz),
      x: sub.map(() => labelOf(mz)),
      y: sub.map((r) => r[metric]),
      text: sub.map((r) => r.sample_name),
      boxpoints: "all",
      jitter: 0.35,
      pointpos: 0,
      marker: { color, size: 8, line: { width: 2, color: dark ? "#111a2e" : "#ffffff" } },
      line: { color, width: 2 },
      fillcolor: "rgba(0,0,0,0)",
      hovertemplate: "%{text}: %{y:.5f}<extra></extra>",
    } as Data;
  });

  const values = rows.map((r) => r[metric]).filter((v): v is number => v != null);
  const top = Math.max(...values);
  const span = top - Math.min(...values) || Math.abs(top) || 1;
  const shapes: Partial<Shape>[] = [];
  const annotations: Partial<Annotations>[] = [];
  compare.pairs
    .filter((p) => p.reject)
    .forEach((p, level) => {
      const y = top + span * (0.12 + 0.12 * level);
      // Path coordinates on a category axis are category positions (0, 1, ...), not names.
      const a = order.indexOf(p.group_a);
      const b = order.indexOf(p.group_b);
      if (a < 0 || b < 0) return;
      shapes.push({
        type: "path",
        xref: "x",
        yref: "y",
        path: `M ${a},${y - span * 0.03} L ${a},${y} L ${b},${y} L ${b},${y - span * 0.03}`,
        line: { width: 1, color: dark ? "#a3b1c6" : "#475569" },
      });
      annotations.push({
        x: labelOf(p.group_b),
        y,
        xref: "x",
        yref: "y",
        text: stars(p.adjusted_p_value),
        showarrow: false,
        xanchor: "left",
        xshift: 4,
        font: { size: 13 },
      });
    });

  return {
    data,
    layout: {
      showlegend: order.length > 1,
      xaxis: { type: "category", title: { text: "Target" } },
      yaxis: { title: { text: METRIC_LABEL[metric] } },
      shapes: shapes as Layout["shapes"],
      annotations,
      hovermode: "closest",
    },
  };
}
