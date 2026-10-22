import type { Data, Layout, Shape } from "plotly.js";
import { AlertTriangle, ArrowRightLeft, CheckCircle2, Loader2, X, XCircle } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import { api, type CandidateDetail, type Criterion, type DiscoveryCandidate } from "../api/client";
import { Plot } from "../components/Plot";
import { Banner } from "../components/ui";
import { fmtMz } from "../lib/format";
import { SERIES_DARK, SERIES_LIGHT } from "../lib/palette";
import { useDarkMode } from "../lib/theme";
import { useFetch } from "../lib/useFetch";
import { useStore } from "../store";
import { prettyKey } from "./DiscoveryView";

const CRITERION_LABEL: Record<string, string> = {
  delta_rt: "RT difference (min)",
  rt_resolution: "RT resolution Rs",
  mobility_rel_pct: "Mobility difference (%)",
};

/** Species A = palette slot 0, B = slot 1, the same on every plot of the viewer. */
function speciesColours(dark: boolean): Record<string, string> {
  const p = dark ? SERIES_DARK : SERIES_LIGHT;
  return { A: p[0], B: p[1] };
}

function maxOf(v: number[]): number {
  let m = 0;
  for (const x of v) if (x > m) m = x;
  return m || 1;
}

/** Vertical sticks (m/z, intensity) as one scatter trace with gaps between sticks. */
function sticks(mz: number[], y: number[], sign = 1): { x: (number | null)[]; y: (number | null)[] } {
  const x: (number | null)[] = [];
  const yy: (number | null)[] = [];
  mz.forEach((m, i) => {
    x.push(m, m, null);
    yy.push(0, sign * y[i], null);
  });
  return { x, y: yy };
}

export function CandidateViewer({ runId, candidateId, onClose }: { runId: string; candidateId: number; onClose: () => void }) {
  const dark = useDarkMode();
  const ref = useRef<HTMLElement>(null);
  // Scroll only the main pane: scrollIntoView would also shift the overflow-hidden app shell.
  useEffect(() => {
    const el = ref.current;
    const pane = el?.closest<HTMLElement>(".main");
    if (el && pane) pane.scrollTo({ top: pane.scrollTop + el.getBoundingClientRect().top - pane.getBoundingClientRect().top - 12, behavior: "smooth" });
  }, [candidateId]);
  const res = useFetch<CandidateDetail>(`${runId}/${candidateId}`, (signal) =>
    api.get(`/discovery/runs/${runId}/candidates/${candidateId}`, signal),
  );

  return (
    <section className="card" aria-labelledby="cv-h" ref={ref}>
      <div className="card-head">
        <h2 id="cv-h">Candidate {candidateId}</h2>
        <button className="btn btn-ghost btn-icon btn-sm" aria-label="Close candidate" onClick={onClose}>
          <X size={15} />
        </button>
      </div>
      <div className="card-body stack" style={{ gap: 16 }}>
        {res.loading && !res.data && (
          <p className="muted row"><Loader2 size={14} className="spin" aria-hidden /> Loading traces…</p>
        )}
        {res.error && <Banner kind="err" title="Could not load the candidate">{res.error.message}</Banner>}
        {res.data && <Detail d={res.data} dark={dark} runId={runId} />}
      </div>
    </section>
  );
}

function Detail({ d, dark, runId }: { d: CandidateDetail; dark: boolean; runId: string }) {
  const c = d.candidate;
  const col = speciesColours(dark);
  return (
    <>
      <Summary c={c} />
      {d.note && <Banner kind="warn" title="Traces unavailable">{d.note}</Banner>}
      {d.detail && <Traces detail={d.detail} col={col} dark={dark} c={c} />}
      <div className="grid-2">
        <Criteria c={c} />
        <Evidence d={d} />
      </div>
      {d.detail?.mirror && <Mirror mirror={d.detail.mirror} col={col} dark={dark} />}
      {d.msms.differential_fragments.length > 0 && <Differential rows={d.msms.differential_fragments} />}
      <Handoff c={c} runId={runId} />
      {d.detail && (
        <p className="faint" style={{ fontSize: 12 }}>
          Measured: {d.detail.labels.measured.join(", ")}. Calculated: {d.detail.labels.calculated.join(", ")}.
        </p>
      )}
    </>
  );
}

function Summary({ c }: { c: DiscoveryCandidate }) {
  const fa = c.relative_signal_fraction_a;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row" style={{ gap: 8 }}>
        <span className="badge">{c.classification}</span>
        <span className={`badge ${c.status === "excluded" ? "badge-err" : "badge-info"}`}>
          {c.status === "excluded" ? "Excluded" : c.category}
        </span>
        <span className="badge">route: {c.route}</span>
        <span className="badge">{c.sequence_status}</span>
      </div>
      <div className="tiles">
        <Fact label="Sequence" value={prettyKey(c.mod_key)} mono />
        <Fact label="Monoisotopic m/z" value={`${fmtMz(c.mono_mz)} (${c.charge}+)`} />
        <Fact label="Neutral mass" value={`${c.neutral_mass.toFixed(4)} Da`} />
        <Fact label="ΔRT" value={c.delta_rt != null ? `${c.delta_rt.toFixed(2)} min` : "–"} sub={c.rt_resolution != null ? `Rs ${c.rt_resolution.toFixed(2)} · ${c.rt_separation}` : undefined} />
        <Fact
          label="Δ mobility"
          value={c.mobility_rel_diff_pct != null ? `${c.mobility_rel_diff_pct.toFixed(2)}%` : "–"}
          sub={c.delta_mobility != null ? `${c.delta_mobility.toFixed(4)} · ref. ${c.mobility_reference}` : undefined}
        />
        {fa != null && c.species_b && (
          <Fact label="Relative signal A : B" value={`${(fa * 100).toFixed(0)} : ${((1 - fa) * 100).toFixed(0)}`} sub="LC-MS signal, not molar abundance" />
        )}
      </div>
      {c.exclusion_reasons.length > 0 && (
        <Banner kind="err" title="Excluded (kept for review)">{c.exclusion_reasons.join(" · ")}</Banner>
      )}
    </div>
  );
}

function Fact({ label, value, sub, mono }: { label: string; value: string; sub?: string; mono?: boolean }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className={mono ? "mono" : ""} style={{ fontSize: mono ? 14 : 18, fontWeight: 600, wordBreak: "break-all" }}>{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

type Detail = NonNullable<CandidateDetail["detail"]>;

function Traces({ detail, col, dark, c }: { detail: Detail; col: Record<string, string>; dark: boolean; c: DiscoveryCandidate }) {
  const coord = detail.coordinate === "1/K0" ? "1/K0 (V·s/cm²)" : detail.coordinate ?? "mobility";

  const eic = useMemo(() => {
    const shapes: Partial<Shape>[] = [];
    for (const s of detail.species) {
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: s.rt_start, x1: s.rt_end, y0: 0, y1: 1,
                    fillcolor: col[s.name], opacity: 0.12, line: { width: 0 }, layer: "below" });
      shapes.push({ type: "line", xref: "x", yref: "paper", x0: s.rt_apex, x1: s.rt_apex, y0: 0, y1: 1,
                    line: { color: col[s.name], width: 1.5, dash: s.name === "A" ? "solid" : "dash" } });
    }
    const data: Data[] = [{ x: detail.chromatogram.rt, y: detail.chromatogram.intensity, type: "scatter", mode: "lines",
                            name: `EIC (${detail.chromatogram.isotopes_summed} isotopes, ±${detail.chromatogram.ppm} ppm)`,
                            line: { color: dark ? "#c8ccd2" : "#3b4048", width: 1.4 } }];
    for (const s of detail.species)
      data.push({ x: [s.rt_apex], y: [s.height], type: "scatter", mode: "markers", name: `Species ${s.name}`,
                  marker: { color: col[s.name], size: 9, symbol: s.name === "A" ? "circle" : "diamond" },
                  hovertemplate: `Species ${s.name}<br>apex %{x:.2f} min<extra></extra>` });
    const layout: Partial<Layout> = { shapes: shapes as Layout["shapes"], xaxis: { title: { text: "Retention time (min)" } },
                                      yaxis: { title: { text: "Intensity" } }, margin: { t: 28 } };
    return { data, layout };
  }, [detail, col, dark]);

  const mob = useMemo(() => {
    // Species cut from the same RT peak (IM-separated) share one mobilogram: draw it once.
    const shared = detail.species.length === 2 && detail.species[0].rt_peak === detail.species[1].rt_peak;
    const traces = shared ? detail.mobilograms.slice(0, 1) : detail.mobilograms;
    const data: Data[] = traces.map((m) => {
      const top = maxOf(m.y);
      return { x: m.x, y: m.y.map((v) => v / top), type: "scatter", mode: "lines",
               name: shared ? "Mobilogram (shared RT window)" : `Species ${m.species}`,
               line: { color: shared ? (dark ? "#c8ccd2" : "#3b4048") : col[m.species], width: 1.8,
                       dash: shared || m.species === "A" ? "solid" : "dash" } } as Data;
    });
    for (const m of detail.mobilograms)
      if (m.centre != null)
        data.push({ x: [m.centre, m.centre], y: [0, 1.05], type: "scatter", mode: "lines", name: `Centre ${m.species}`,
                    line: { color: col[m.species], width: 1.5, dash: m.species === "A" ? "dot" : "dash" },
                    hovertemplate: `Species ${m.species} centre %{x:.4f}<extra></extra>` } as Data);
    const layout: Partial<Layout> = { xaxis: { title: { text: coord } },
                                      yaxis: { title: { text: "Normalized intensity" } }, margin: { t: 28 } };
    return { data, layout };
  }, [detail, col, coord, dark]);

  const map = useMemo(() => {
    if (!detail.map) return null;
    const data: Data[] = [
      { x: detail.map.rt, y: detail.map.mobility, z: detail.map.z.map((r) => r.map(Math.sqrt)), type: "heatmap", showscale: false,
        // Neutral ink, so the species colours stay reserved for A and B; empty bins stay transparent.
        colorscale: dark
          ? [[0, "rgba(0,0,0,0)"], [1e-9, "#2a2e35"], [1, "#e8eaed"]]
          : [[0, "rgba(0,0,0,0)"], [1e-9, "#e6e8eb"], [1, "#1f2329"]], hovertemplate: "RT %{x:.2f}<br>mobility %{y:.4f}<br>√intensity %{z:.3s}<extra></extra>" } as Data,
      ...detail.species
        .filter((s) => s.mobility != null)
        .map((s) => ({ x: [s.rt_apex], y: [s.mobility], type: "scatter", mode: "markers", name: `Species ${s.name}`,
                       marker: { color: col[s.name], size: 10, symbol: s.name === "A" ? "circle-open" : "diamond-open", line: { width: 2 } } }) as Data),
    ];
    const layout: Partial<Layout> = { xaxis: { title: { text: "Retention time (min)" } }, yaxis: { title: { text: coord } },
                                      margin: { t: 28 }, showlegend: true };
    return { data, layout };
  }, [detail, col, dark, coord]);

  const ms1 = useMemo(() => {
    const data: Data[] = [];
    for (const s of detail.ms1) {
      const top = maxOf(s.intensity);
      const st = sticks(s.mz, s.intensity.map((v) => v / top));
      data.push({ x: st.x, y: st.y, type: "scatter", mode: "lines", name: `Species ${s.species} MS1`, connectgaps: false,
                  line: { color: col[s.species], width: 1.4 } } as Data);
      const atop = maxOf(s.averagine_relative);
      data.push({ x: s.averagine_mz, y: s.averagine_relative.map((v) => v / atop), type: "scatter", mode: "markers",
                  name: `Averagine (${s.species})`, marker: { color: col[s.species], symbol: "x-thin-open", size: 10, line: { width: 2 } } } as Data);
    }
    const layout: Partial<Layout> = { xaxis: { title: { text: "m/z" } }, yaxis: { title: { text: "Relative intensity" } }, margin: { t: 28 } };
    return { data, layout };
  }, [detail, col]);

  return (
    <div className="stack" style={{ gap: 12 }}>
      <Plot className="plot-eic" dark={dark} data={eic.data} layout={eic.layout}
            ariaLabel={`Isotope-summed chromatogram at m/z ${fmtMz(c.mono_mz)} with ${detail.species.length} species marked`} />
      <div className="grid-2">
        <Plot className="plot-mob" dark={dark} data={mob.data} layout={mob.layout}
              ariaLabel={`Mobilograms of ${detail.mobilograms.length} species`} />
        {map ? (
          <Plot className="plot-mob" dark={dark} data={map.data} layout={map.layout} ariaLabel="Retention time by mobility intensity map" />
        ) : (
          <p className="muted">No mobility data to map.</p>
        )}
      </div>
      <Plot className="plot-mob" dark={dark} data={ms1.data} layout={ms1.layout}
            ariaLabel="MS1 isotope envelopes compared with the averagine pattern" />
    </div>
  );
}

function Criteria({ c }: { c: DiscoveryCandidate }) {
  const entries = Object.entries(c.criteria);
  return (
    <div className="stack" style={{ gap: 8 }}>
      <h3 style={{ fontSize: 14 }}>Criteria</h3>
      {entries.length === 0 ? (
        <p className="muted">No pairwise criteria (single observed species).</p>
      ) : (
        <table className="data">
          <thead><tr><th>Criterion</th><th className="r">Value</th><th className="r">Threshold</th><th>Met</th></tr></thead>
          <tbody>
            {entries.map(([k, v]: [string, Criterion]) => (
              <tr key={k}>
                <td>{CRITERION_LABEL[k] ?? k}</td>
                <td className="r num">{v.value != null ? v.value.toFixed(3) : "–"}</td>
                <td className="r num">{v.threshold}</td>
                <td>{v.met ? <CheckCircle2 size={14} aria-label="met" style={{ color: "var(--success)" }} /> : <XCircle size={14} aria-label="not met" style={{ color: "var(--text-faint)" }} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {c.flags.length > 0 && (
        <>
          <h3 style={{ fontSize: 14 }}>Flags</h3>
          <ul className="stack" style={{ gap: 4, paddingLeft: 0, listStyle: "none" }}>
            {c.flags.map((f) => (
              <li key={f} className="row" style={{ gap: 6, flexWrap: "nowrap", fontSize: 13 }}>
                <AlertTriangle size={13} aria-hidden style={{ color: "var(--accent)", flex: "none" }} /> {f}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Evidence({ d }: { d: CandidateDetail }) {
  const { a, b, similarity, annotation } = d.msms;
  const c = d.candidate;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <h3 style={{ fontSize: 14 }}>MS/MS evidence</h3>
      <p style={{ fontSize: 13 }}>
        Status: <strong>{c.msms_status ?? "not assessed"}</strong>
      </p>
      <table className="data">
        <thead><tr><th>Species</th><th className="r">Spectra</th><th className="r">Unambiguous</th><th className="r">Replicate similarity</th></tr></thead>
        <tbody>
          {(["A", "B"] as const).map((k) => {
            const s = k === "A" ? a : b;
            if (!s) return <tr key={k}><td>{k}</td><td colSpan={3} className="faint">no MS/MS assigned</td></tr>;
            return (
              <tr key={k}>
                <td>{k}</td>
                <td className="r num">{s.n_assigned}</td>
                <td className="r num">{s.n_unambiguous}</td>
                <td className="r num">{s.within_similarity_median != null ? s.within_similarity_median.toFixed(2) : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {similarity && (
        <p style={{ fontSize: 13 }}>
          A vs B cosine <strong className="num">{similarity.cosine?.toFixed(3) ?? "–"}</strong> · spectral angle{" "}
          <span className="num">{similarity.spectral_angle?.toFixed(3) ?? "–"}</span> · {similarity.n_matched} matched peaks
        </p>
      )}
      {Object.keys(annotation).length > 0 && (
        <table className="data">
          <thead><tr><th>Sequence</th><th className="r">A: b/y explained</th><th className="r">B: b/y explained</th></tr></thead>
          <tbody>
            {Object.entries(annotation).map(([key, v]) => (
              <tr key={key}>
                <td className="mono" style={{ fontSize: 12 }}>{prettyKey(key)}</td>
                <td className="r num">{v.a?.fraction != null ? `${(v.a.fraction * 100).toFixed(0)}% (${v.a.n_ions})` : "–"}</td>
                <td className="r num">{v.b?.fraction != null ? `${(v.b.fraction * 100).toFixed(0)}% (${v.b.n_ions})` : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="faint" style={{ fontSize: 12 }}>
        Similar MS/MS supports, but does not prove, a shared sequence; different MS/MS does not rule out isomerization.
        Identity is never transferred from one species to the other.
      </p>
    </div>
  );
}

function Mirror({ mirror, col, dark }: { mirror: NonNullable<Detail["mirror"]>; col: Record<string, string>; dark: boolean }) {
  const plot = useMemo(() => {
    const ta = maxOf(mirror.a_intensity);
    const tb = maxOf(mirror.b_intensity);
    const a = sticks(mirror.a_mz, mirror.a_intensity.map((v) => v / ta));
    const b = sticks(mirror.b_mz, mirror.b_intensity.map((v) => v / tb), -1);
    const data: Data[] = [
      { x: a.x, y: a.y, type: "scatter", mode: "lines", name: "Species A", line: { color: col.A, width: 1.3 } } as Data,
      { x: b.x, y: b.y, type: "scatter", mode: "lines", name: "Species B", line: { color: col.B, width: 1.3 } } as Data,
    ];
    const layout: Partial<Layout> = { xaxis: { title: { text: "m/z" } },
                                      yaxis: { title: { text: "Relative intensity (A up, B down)" }, range: [-1.08, 1.08] },
                                      margin: { t: 28 } };
    return { data, layout };
  }, [mirror, col]);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <h3 style={{ fontSize: 14 }}>MS/MS mirror plot</h3>
      <Plot className="plot-mob" dark={dark} data={plot.data} layout={plot.layout}
            ariaLabel={`MS/MS mirror plot, ${mirror.matches.length} matched fragment peaks`} />
    </div>
  );
}

function Differential({ rows }: { rows: CandidateDetail["msms"]["differential_fragments"] }) {
  return (
    <div className="stack" style={{ gap: 6 }}>
      <h3 style={{ fontSize: 14 }}>Most different fragments</h3>
      <div className="table-wrap" style={{ maxHeight: 240 }}>
        <table className="data">
          <thead><tr><th className="r">m/z</th><th className="r">A (rel.)</th><th className="r">B (rel.)</th><th className="r">log₂ A/B</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="r num">{fmtMz(r.mz)}</td>
                <td className="r num">{r.a.toFixed(3)}</td>
                <td className="r num">{r.b.toFixed(3)}</td>
                <td className="r num">{r.log_ratio == null ? (r.a > 0 ? "A only" : "B only") : r.log_ratio.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Handoff({ c, runId }: { c: DiscoveryCandidate; runId: string }) {
  const datasets = useStore((s) => s.datasets);
  const run = useFetch<{ meta: { dataset: { source_path?: string; name?: string } } }>(`handoff-${runId}`, (signal) =>
    api.get(`/discovery/runs/${runId}`, signal),
  );
  const sequence = c.mod_key?.split("|")[0] ?? "";
  const modified = !!c.mod_key?.split("|")[1];
  const ready = !!c.species_b;

  const go = () => {
    const st = useStore.getState();
    const src = run.data?.meta.dataset.source_path;
    const ds = Object.values(datasets).find((d) => d.path === src || d.local_path === src);
    if (ds) {
      st.toggleSelected(ds.id, true);
      st.setActiveSample(ds.id);
    }
    const species = [c.species_a, c.species_b!];
    const warning = st.handoffToLocalization(
      [{ mz: c.mono_mz, label: `precursor ${c.charge}+` }],
      species.map((s) => ({ start: Math.round(s.rt_start * 100) / 100, end: Math.round(s.rt_end * 100) / 100 })),
      sequence,
      [1, 2],
    );
    if (!ds) st.notify("err", "The run's dataset is no longer registered; select it on the Datasets screen.");
    else if (warning) st.notify("err", warning);
    else st.notify("ok", "Candidate sent to localization: compute fragments on the Targets screen");
  };

  return (
    <div className="row" style={{ gap: 10, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
      <button className="btn btn-primary" disabled={!ready} onClick={go}>
        <ArrowRightLeft size={14} aria-hidden /> Localize with CID-TIMS
      </button>
      <span className="muted" style={{ fontSize: 13, flex: 1 }}>
        {ready
          ? `Sets two RT windows (species A and B), the precursor target${sequence ? ` and the peptide ${sequence}` : ""}, then opens the localization workflow.`
          : "Localization needs two observed species."}
        {ready && c.species_b && c.species_a.rt_peak === c.species_b.rt_peak &&
          " Both species elute in the same RT window, so the two windows coincide; tell them apart by mobility."}
        {ready && modified && " The fragment calculator uses the unmodified sequence; add modified fragment m/z by hand."}
      </span>
    </div>
  );
}
