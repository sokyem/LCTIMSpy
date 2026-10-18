import type { Annotations, Data, Layout, Shape } from "plotly.js";
import { Crosshair, Loader2, MousePointerSquareDashed, Plus, Sparkles, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, ApiError, download, type MobilogramSeries, type Peak, type Series } from "../api/client";
import { Plot } from "../components/Plot";
import { Banner, Empty, NumberInput, Segmented } from "../components/ui";
import { fmtMz } from "../lib/format";
import { MAX_WINDOWS, seriesColor, WINDOW_DASH, windowName } from "../lib/palette";
import { breakGaps, normalize, rangeFromSelection, round, type RtWindow, windowEditsFromRelayout } from "../lib/relayout";
import { cssVar, useDarkMode } from "../lib/theme";
import { useFetch } from "../lib/useFetch";
import { enabledTargets, type Target, useReadySamples, useStore } from "../store";

const DASH_CSS: Record<string, string> = { solid: "solid", dash: "dashed", dot: "dotted", dashdot: "dashed" };

export function WindowsView() {
  const dark = useDarkMode();
  const samples = useReadySamples();
  const activeSample = useStore((s) => s.activeSample);
  const setActiveSample = useStore((s) => s.setActiveSample);
  const targets = useStore((s) => s.targets);
  const windows = useStore((s) => s.windows);
  const armed = useStore((s) => s.armed);
  const offsets = useStore((s) => s.offsets);
  const query = useStore((s) => s.query);
  const { setWindow, setWindows, removeWindow, arm, fillArmed, setOffset, notify, setView } = useStore.getState();

  const active = samples.find((d) => d.id === activeSample) ?? samples[0];
  const tgs = useMemo(() => enabledTargets(targets), [targets]);
  const offset = (active && offsets[active.id]) || 0;
  const shown = useMemo(
    () => windows.map((w) => ({ start: round(w.start + offset), end: round(w.end + offset) })),
    [windows, offset],
  );
  const [normalized, setNormalized] = useState(true);
  const [detecting, setDetecting] = useState(false);

  // ---- data ---------------------------------------------------------------------
  const apiTargets = tgs.map((t) => ({ mz: t.mz, label: t.label }));
  const eicReq = active && tgs.length
    ? { dataset_ids: [active.id], targets: apiTargets, tolerance: query.tolerance, max_points: 2500 }
    : null;
  const eic = useFetch<{ series: Series[] }>(
    eicReq && JSON.stringify(eicReq),
    (signal) => api.post("/signals/chromatograms", eicReq, signal),
    120,
  );
  const mobReq = active && tgs.length && shown.length
    ? { dataset_ids: [active.id], targets: apiTargets, windows: shown, query }
    : null;
  // Debounced so dragging a window edge does not fire a query per pixel.
  const mob = useFetch<{ series: MobilogramSeries[] }>(
    mobReq && JSON.stringify(mobReq),
    (signal) => api.post("/signals/mobilograms", mobReq, signal),
    300,
  );

  const detect = useCallback(async () => {
    if (!active || !tgs.length) return;
    setDetecting(true);
    try {
      const { peaks } = await api.post<{ peaks: Peak[] }>("/chromatograms/peaks", {
        dataset_id: active.id,
        targets: tgs.map((t) => ({ mz: t.mz })),
        tolerance: query.tolerance,
        n: 2,
      });
      if (peaks.length < 2) {
        notify("err", `Found ${peaks.length} peak${peaks.length === 1 ? "" : "s"}; drag across the chromatogram to set windows.`);
        if (peaks.length === 0) return;
      }
      setWindows(peaks.map((p) => ({ start: round(p.start_rt - offset), end: round(p.end_rt - offset) })));
    } catch (e) {
      notify("err", `Peak detection failed: ${e instanceof ApiError ? e.message : String(e)}`);
    } finally {
      setDetecting(false);
    }
  }, [active, tgs, query.tolerance, offset, notify, setWindows]);

  // The isomer workflow is definitionally two peaks: land on them.
  const autoRan = useRef(false);
  useEffect(() => {
    if (!autoRan.current && windows.length === 0 && eic.data?.series.some((s) => s.y.some((v) => v > 0))) {
      autoRan.current = true;
      void detect();
    }
  }, [eic.data, windows.length, detect]);

  // ---- chart model ----------------------------------------------------------
  const slotOf = (mz: number) => tgs.find((t) => Math.abs(t.mz - mz) < 1e-9)?.slot ?? null;
  const nameOf = (s: { label?: string | null; target_mz: number }) => s.label || `m/z ${fmtMz(s.target_mz)}`;

  const eicData = useMemo<Data[]>(
    () =>
      (eic.data?.series ?? []).map((s) => ({
        type: "scatter",
        mode: "lines",
        x: s.x,
        y: s.y,
        name: nameOf(s),
        line: { color: seriesColor(slotOf(s.target_mz), dark) ?? undefined, width: 2 },
        hovertemplate: `${nameOf(s)}<br>RT %{x:.3f} min<br>%{y:,.0f}<extra></extra>`,
      })),
    [eic.data, dark, tgs],
  );

  const eicLayout = useMemo<Partial<Layout>>(() => {
    const band = cssVar("--band");
    const edge = cssVar("--band-edge");
    const text = cssVar("--text");
    const shapes: Partial<Shape>[] = shown.map((w, i) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: w.start,
      x1: w.end,
      y0: 0,
      y1: 1,
      fillcolor: band,
      line: { color: edge, width: 1.5, dash: WINDOW_DASH[i] },
      layer: "below",
      // Bottom-left: the chromatogram is near baseline at a window's edge, while
      // peak apexes (and their direct labels) are at the top.
      label: { text: windowName(i), textposition: "bottom left", font: { color: text, size: 12 } },
    }));
    // Direct labels at each apex when there are few series (legend stays too).
    const series = eic.data?.series ?? [];
    const annotations: Partial<Annotations>[] =
      series.length <= 4
        ? series.map((s) => {
            const i = s.y.reduce((best, v, j, arr) => (v > arr[best] ? j : best), 0);
            return {
              x: s.x[i],
              y: s.y[i],
              text: nameOf(s),
              showarrow: false,
              yshift: 10,
              font: { color: text, size: 11 },
            };
          })
        : [];
    return {
      dragmode: "select",
      selectdirection: "h",
      selections: [],
      shapes: shapes as Layout["shapes"],
      annotations,
      xaxis: { title: { text: "Retention time (min)" } },
      yaxis: { title: { text: "Intensity" }, rangemode: "tozero" },
      hovermode: "x unified",
      margin: { t: 8, b: 40 },
      showlegend: series.length > 1,
      uirevision: active?.id,
      // A new value makes Plotly apply our shapes rather than keep its own edit state.
      editrevision: JSON.stringify(shown),
    } as Partial<Layout>;
  }, [shown, eic.data, dark, active?.id]);

  // While a window is armed, bands are not editable: their drag handles would
  // otherwise capture a redraw gesture that starts on or near the old band.
  const eicConfig = useMemo(() => ({ edits: { shapePosition: armed == null } }), [armed]);

  const mobData = useMemo<Data[]>(
    () =>
      (mob.data?.series ?? []).map((s) => ({
        type: "scatter",
        mode: "lines",
        // Bins without signal are absent: break the line there, don't bridge it.
        ...breakGaps(s.x, normalized ? normalize(s.y) : s.y),
        connectgaps: false,
        name: `${nameOf(s)} · ${windowName(s.window)}`,
        line: {
          color: seriesColor(slotOf(s.target_mz), dark) ?? undefined,
          width: 2,
          dash: WINDOW_DASH[s.window % WINDOW_DASH.length],
        },
        hovertemplate: `${nameOf(s)} · ${windowName(s.window)}<br>1/K0 %{x:.4f}<br>%{y:.3~g}<extra></extra>`,
      })),
    [mob.data, normalized, dark, tgs],
  );
  const mobLayout = useMemo<Partial<Layout>>(
    () => ({
      xaxis: { title: { text: "Mobility 1/K0 (V·s/cm²)" } },
      yaxis: { title: { text: normalized ? "Normalized intensity" : "Intensity" }, rangemode: "tozero" },
      hovermode: "closest",
      margin: { t: 8, b: 40 },
      showlegend: true,
      uirevision: active?.id,
    }),
    [normalized, active?.id],
  );

  // ---- interactions ---------------------------------------------------------
  const onSelected = (e: unknown) => {
    const r = rangeFromSelection(e);
    if (!r) return;
    if (armed == null && windows.length >= MAX_WINDOWS) {
      notify("err", `All ${MAX_WINDOWS} windows are set. Click "Redraw" on one to replace it.`);
      return;
    }
    fillArmed({ start: round(r.start - offset), end: round(r.end - offset) });
  };
  const onRelayout = (payload: Record<string, unknown>) => {
    for (const [i, w] of windowEditsFromRelayout(payload, shown)) {
      setWindow(i, { start: round(w.start - offset), end: round(w.end - offset) });
    }
  };
  const editBound = (i: number, which: keyof RtWindow, value: number) => {
    const w = { ...windows[i], [which]: round(value - offset) };
    if (w.end > w.start) setWindow(i, w);
    else notify("err", "A window's end must be after its start.");
  };

  // ---- render -----------------------------------------------------------------
  if (!samples.length || !tgs.length) {
    return (
      <div className="view">
        <h1>RT windows</h1>
        <div className="card">
          <Empty icon={<Crosshair size={32} aria-hidden />} title={!samples.length ? "No extracted datasets selected" : "No targets plotted"}>
            <p>{!samples.length ? "Select extracted datasets first." : "Choose at least one target to plot."}</p>
            <button className="btn btn-primary" style={{ marginTop: 8 }} onClick={() => setView(!samples.length ? "datasets" : "targets")}>
              Go to {!samples.length ? "Datasets" : "Targets"}
            </button>
          </Empty>
        </div>
      </div>
    );
  }

  const exportReq = { dataset_ids: [active.id], targets: apiTargets, tolerance: query.tolerance };
  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>RT windows</h1>
          <p className="muted">Choose the two elution windows of the isomers. The mobilograms update as you adjust them.</p>
        </div>
        <button className="btn btn-primary" disabled={windows.length < 2} onClick={() => setView("results")}>
          Next: results
        </button>
      </div>

      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row">
          <span className="muted" style={{ fontSize: 13, fontWeight: 600 }}>Sample</span>
          <Segmented
            label="Sample shown"
            value={active.id}
            options={samples.map((d) => ({ value: d.id, label: d.name }))}
            onChange={setActiveSample}
          />
        </div>
        <label className="row" style={{ gap: 6 }} title="Shift this sample's windows to follow retention-time drift">
          <span className="muted" style={{ fontSize: 13, fontWeight: 600 }}>RT offset for {active.name}</span>
          <NumberInput value={offset} step={0.01} label={`RT offset for ${active.name} (min)`} width={84} onCommit={(v) => setOffset(active.id, round(v))} />
          <span className="faint" style={{ fontSize: 12 }}>min</span>
        </label>
      </div>

      <section className="card" aria-labelledby="eic-h">
        <div className="card-head">
          <div className="row">
            <h2 id="eic-h">Extracted-ion chromatogram</h2>
            {eic.loading && <Loader2 size={14} className="spin faint" aria-label="Loading" />}
          </div>
          <div className="row">
            <button className="btn btn-sm" onClick={() => void detect()} disabled={detecting}>
              {detecting ? <Loader2 size={13} className="spin" aria-hidden /> : <Sparkles size={13} aria-hidden />} Detect 2 peaks
            </button>
            <button className="btn btn-sm" onClick={() => void download("chromatograms", "csv", exportReq).catch((e) => notify("err", String(e)))}>
              Export CSV
            </button>
          </div>
        </div>
        <div className="card-body stack" style={{ gap: 8 }}>
          {eic.error && <Banner kind="err" title="Could not load chromatograms">{eic.error.message}</Banner>}
          <Plot
            className="plot-eic"
            dark={dark}
            data={eicData}
            layout={eicLayout}
            config={eicConfig}
            onSelected={onSelected}
            onRelayout={onRelayout}
            ariaLabel={`Extracted-ion chromatograms for ${active.name}, ${eicData.length} targets, ${shown.length} RT windows`}
          />
          <div className="plot-hint">
            <MousePointerSquareDashed size={14} aria-hidden />
            {armed != null
              ? `Drag across the chromatogram to set ${windowName(armed)}.`
              : "Drag a window to move it, or drag its edges to resize. Double-click to reset zoom."}
          </div>
          <div className="windows" aria-label="RT windows">
            {shown.map((w, i) => (
              <div className="wchip" key={i} data-armed={armed === i}>
                <span className="dash-sample" style={{ borderTopStyle: DASH_CSS[WINDOW_DASH[i]] as "solid" }} aria-hidden />
                <span className="wchip-label">{windowName(i)}</span>
                <NumberInput value={w.start} label={`${windowName(i)} start (min)`} onCommit={(v) => editBound(i, "start", v)} />
                <span className="faint">–</span>
                <NumberInput value={w.end} label={`${windowName(i)} end (min)`} onCommit={(v) => editBound(i, "end", v)} />
                <span className="faint" style={{ fontSize: 12 }}>min</span>
                <button className="btn btn-ghost btn-sm" onClick={() => arm(i)} aria-pressed={armed === i} title="Redraw this window by dragging">
                  Redraw
                </button>
                <button className="btn btn-ghost btn-icon btn-sm" onClick={() => removeWindow(i)} aria-label={`Remove ${windowName(i)}`}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {armed === shown.length && (
              <div className="wchip" data-armed="true">
                <span className="wchip-label">{windowName(shown.length)}</span>
                <span className="faint" style={{ fontSize: 13 }}>drag on the chart</span>
              </div>
            )}
            {shown.length < MAX_WINDOWS && armed !== shown.length && (
              <button className="btn btn-sm" onClick={() => arm(shown.length)}>
                <Plus size={13} aria-hidden /> Add window
              </button>
            )}
          </div>
        </div>
      </section>

      <section className="card" aria-labelledby="mob-h">
        <div className="card-head">
          <div className="row">
            <h2 id="mob-h">Mobilograms in each window</h2>
            {mob.loading && <Loader2 size={14} className="spin faint" aria-label="Loading" />}
          </div>
          <Segmented
            label="Mobilogram scale"
            value={normalized ? "norm" : "raw"}
            options={[
              { value: "norm", label: "Normalized" },
              { value: "raw", label: "Intensity" },
            ]}
            onChange={(v) => setNormalized(v === "norm")}
          />
        </div>
        <div className="card-body">
          {mob.error && <Banner kind="err" title="Could not load mobilograms">{mob.error.message}</Banner>}
          {shown.length === 0 ? (
            <Empty icon={<Crosshair size={28} aria-hidden />} title="No windows yet">
              Drag across the chromatogram above, or use Detect 2 peaks.
            </Empty>
          ) : (
            <Plot
              className="plot-mob"
              dark={dark}
              data={mobData}
              layout={mobLayout}
              ariaLabel={`Mobilograms for ${active.name} in ${shown.length} windows`}
            />
          )}
          <MissingNote series={mob.data?.series ?? []} targets={tgs} windows={shown.length} />
        </div>
      </section>
    </div>
  );
}

function MissingNote({ series, targets, windows }: { series: MobilogramSeries[]; targets: Target[]; windows: number }) {
  if (!series.length && !windows) return null;
  const missing: string[] = [];
  for (const t of targets) {
    for (let w = 0; w < windows; w++) {
      if (!series.some((s) => Math.abs(s.target_mz - t.mz) < 1e-9 && s.window === w)) {
        missing.push(`${t.label || fmtMz(t.mz)} in ${windowName(w)}`);
      }
    }
  }
  if (!missing.length) return null;
  return (
    <p className="faint" style={{ fontSize: 12, marginTop: 8 }}>
      No signal for {missing.slice(0, 6).join(", ")}
      {missing.length > 6 ? ` and ${missing.length - 6} more` : ""}.
    </p>
  );
}
