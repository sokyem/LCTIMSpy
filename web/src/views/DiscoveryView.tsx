import { AlertTriangle, Download, FileSearch, FileText, Loader2, Play, Radar, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";

import {
  api,
  ApiError,
  downloadGet,
  type DiscoveryCandidate,
  type DiscoveryRun,
  type DiscoveryRunSummary,
  type Job,
} from "../api/client";
import { FsBrowser } from "../components/FsBrowser";
import { Banner, Empty, NumberInput } from "../components/ui";
import { useFetch } from "../lib/useFetch";
import { useStore } from "../store";
import { CandidateViewer } from "./CandidateViewer";

const CLASS_SHORT: Record<string, string> = {
  "A: RT-separated": "A",
  "B: IM-separated": "B",
  "C: RT- and IM-separated": "C",
  "D: unresolved": "D",
};
const CATEGORY_BADGE: Record<string, string> = {
  "High-confidence putative candidate": "badge-ok",
  "Sequence-supported candidate": "badge-info",
  "Mass-matched candidate": "badge",
  "Unresolved candidate": "badge-warn",
};
const PSM_EXTENSIONS = [".csv", ".tsv", ".txt", ".mzid", ".mztab"];

interface Overrides {
  mass_ppm: number;
  rt_min_delta: number;
  rt_min_resolution: number;
  mobility_min_rel_pct: number;
  mobility_precision_rel: number | null;
  similarity_threshold: number;
  min_snr: number;
}
const DEFAULTS: Overrides = {
  mass_ppm: 5,
  rt_min_delta: 1.0,
  rt_min_resolution: 0.8,
  mobility_min_rel_pct: 2.0,
  mobility_precision_rel: null,
  similarity_threshold: 0.7,
  min_snr: 5,
};

export function prettyKey(key: string | null | undefined): string {
  if (!key) return "unidentified";
  const [seq, mods] = key.split("|");
  return mods ? `${seq} (${mods.replaceAll(";", ", ").replaceAll(":", ": ")})` : seq;
}

interface Preview {
  engine: string;
  total: number;
  with_mobility: number;
  decoys: number;
  runs: { run: string; psms: number }[];
}

export function DiscoveryView() {
  const datasets = useStore((s) => s.datasets);
  const version = useStore((s) => s.discoveryVersion);
  const runId = useStore((s) => s.discoveryRun);
  const { setDiscoveryRun, notify, upsertJob, setView } = useStore.getState();
  const jobs = useStore((s) => s.jobs);

  const extracted = Object.values(datasets).filter((d) => d.cache.status === "cached");
  const [datasetId, setDatasetId] = useState<string>("");
  const [psmPath, setPsmPath] = useState<string | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [choosing, setChoosing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [o, setO] = useState<Overrides>(DEFAULTS);
  const [opened, setOpened] = useState<number | null>(null);

  const runs = useFetch<DiscoveryRunSummary[]>(`runs-${version}`, (signal) => api.get("/discovery/runs", signal));
  const activeId = runId && runs.data?.some((r) => r.id === runId) ? runId : runs.data?.[0]?.id ?? null;
  const run = useFetch<DiscoveryRun>(activeId && `${activeId}-${version}`, (signal) =>
    api.get(`/discovery/runs/${activeId}`, signal),
  );
  const running = Object.values(jobs).filter((j) => j.kind === "discover" && ["queued", "running"].includes(j.state));
  const selected = datasetId || extracted[0]?.id || "";

  const choosePsm = async (paths: string[]) => {
    setPsmPath(paths[0]);
    setPreview(null);
    setPreviewError(null);
    try {
      setPreview(await api.post<Preview>("/discovery/psm-preview", { path: paths[0] }));
    } catch (e) {
      setPreviewError(e instanceof ApiError ? e.message : String(e));
    }
  };

  const start = async () => {
    try {
      const overrides: Record<string, number> = {};
      for (const [k, v] of Object.entries(o)) if (v != null && v !== DEFAULTS[k as keyof Overrides]) overrides[k] = v as number;
      if (o.mobility_precision_rel != null) overrides.mobility_precision_rel = o.mobility_precision_rel;
      const job = await api.post<Job>("/discovery/runs", { dataset_id: selected, psm_path: psmPath, overrides });
      upsertJob(job);
      notify("ok", "Discovery started; it runs in the background");
    } catch (e) {
      notify("err", e instanceof ApiError ? e.message : String(e));
    }
  };

  const remove = async (id: string) => {
    if (!confirm("Delete this discovery run?")) return;
    await api.del(`/discovery/runs/${id}`);
    if (runId === id) setDiscoveryRun(null);
    useStore.getState().bumpDiscovery();
  };

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Isomer discovery</h1>
          <p className="muted">
            Untargeted search for species that share an accurate mass but differ in retention time, ion mobility, or both.
            Every result is a putative candidate.
          </p>
        </div>
      </div>

      <section className="card" aria-labelledby="new-h">
        <div className="card-head">
          <h2 id="new-h">New run</h2>
          <button className="btn btn-ghost btn-sm" onClick={() => setShowSettings((v) => !v)} aria-expanded={showSettings}>
            {showSettings ? "Hide thresholds" : "Thresholds"}
          </button>
        </div>
        <div className="card-body stack" style={{ gap: 14 }}>
          {extracted.length === 0 ? (
            <Banner kind="info" title="No extracted datasets" action={<button className="btn btn-sm" onClick={() => setView("datasets")}>Go to Datasets</button>}>
              Discovery reads the extraction cache. Add and extract a DDA-PASEF .d folder or an mzML file first.
            </Banner>
          ) : (
            <div className="row" style={{ alignItems: "flex-end", gap: 16 }}>
              <label className="field" style={{ minWidth: 220 }}>
                <span>Dataset</span>
                <select className="select" value={selected} onChange={(e) => setDatasetId(e.target.value)}>
                  {extracted.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </label>
              <div className="field" style={{ flex: 1, minWidth: 260 }}>
                <span>Identifications (optional)</span>
                <div className="row" style={{ flexWrap: "nowrap" }}>
                  <button className="btn" onClick={() => setChoosing(true)}>
                    <FileSearch size={14} aria-hidden /> {psmPath ? "Change PSM file" : "Choose PSM file"}
                  </button>
                  {psmPath && (
                    <>
                      <span className="mono faint" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={psmPath}>
                        {psmPath}
                      </span>
                      <button className="btn btn-ghost btn-icon btn-sm" aria-label="Remove PSM file" onClick={() => { setPsmPath(null); setPreview(null); }}>
                        <X size={14} />
                      </button>
                    </>
                  )}
                </div>
                <small>PEAKS, MSFragger psm.tsv, MaxQuant evidence.txt, mzIdentML or mzTab. Without one, only mass-first discovery runs.</small>
              </div>
              <button className="btn btn-primary" disabled={!selected || running.length > 0} onClick={() => void start()}>
                {running.length ? <Loader2 size={14} className="spin" aria-hidden /> : <Play size={14} aria-hidden />}
                {running.length ? "Running…" : "Run discovery"}
              </button>
            </div>
          )}
          {preview && (
            <div className="row" style={{ gap: 8 }}>
              <span className="badge badge-info">{preview.engine}</span>
              <span className="muted" style={{ fontSize: 13 }}>
                {preview.total.toLocaleString()} PSMs · {preview.with_mobility.toLocaleString()} with mobility · {preview.decoys} decoys ·
                runs: {preview.runs.slice(0, 4).map((r) => `${r.run || "(unnamed)"} (${r.psms})`).join(", ")}
                {preview.runs.length > 4 ? "…" : ""}
              </span>
            </div>
          )}
          {previewError && <Banner kind="err" title="Cannot read the PSM file">{previewError}</Banner>}
          {showSettings && <Thresholds value={o} onChange={setO} />}
        </div>
      </section>

      {runs.data && runs.data.length > 0 ? (
        <div className="row" style={{ gap: 8 }} role="tablist" aria-label="Discovery runs">
          {runs.data.map((r) => (
            <span key={r.id} className="row" style={{ gap: 2, flexWrap: "nowrap" }}>
              <button
                role="tab"
                aria-selected={r.id === activeId}
                className={`btn btn-sm ${r.id === activeId ? "btn-primary" : ""}`}
                onClick={() => { setDiscoveryRun(r.id); setOpened(null); }}
              >
                {r.dataset.name} · {new Date(r.created * 1000).toLocaleString()}
              </button>
              <button className="btn btn-ghost btn-icon btn-sm" aria-label={`Delete run ${r.id}`} onClick={() => void remove(r.id)}>
                <Trash2 size={13} />
              </button>
            </span>
          ))}
        </div>
      ) : (
        !runs.loading && (
          <div className="card">
            <Empty icon={<Radar size={32} aria-hidden />} title="No discovery runs yet">
              Choose an extracted dataset and run discovery.
            </Empty>
          </div>
        )
      )}

      {run.error && <Banner kind="err" title="Could not load the run">{run.error.message}</Banner>}
      {run.data && <RunResults run={run.data} opened={opened} onOpen={setOpened} />}
      {choosing && (
        <FsBrowser
          mode="file"
          extensions={PSM_EXTENSIONS}
          title="Choose a PSM export"
          onClose={() => setChoosing(false)}
          onAdd={choosePsm}
        />
      )}
    </div>
  );
}

function Thresholds({ value, onChange }: { value: Overrides; onChange: (o: Overrides) => void }) {
  const set = (k: keyof Overrides) => (v: number) => onChange({ ...value, [k]: v });
  return (
    <div className="stack" style={{ gap: 10, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
      <p className="faint" style={{ fontSize: 12 }}>
        Screening defaults follow the published PepSIFinder workflow (1.0 min; 2% mobility). Neither is universal: set them from your
        chromatographic method and your instrument's measured mobility precision.
      </p>
      <div className="row" style={{ gap: 18, alignItems: "flex-end" }}>
        <Setting label="Mass tolerance (ppm)"><NumberInput value={value.mass_ppm} step={0.5} min={0.1} label="Mass tolerance" width={80} onCommit={set("mass_ppm")} /></Setting>
        <Setting label="ΔRT (min)"><NumberInput value={value.rt_min_delta} step={0.1} min={0} label="Minimum RT difference" width={80} onCommit={set("rt_min_delta")} /></Setting>
        <Setting label="RT resolution Rs ≥"><NumberInput value={value.rt_min_resolution} step={0.1} min={0} label="Minimum RT resolution" width={80} onCommit={set("rt_min_resolution")} /></Setting>
        <Setting label="Mobility difference (%)"><NumberInput value={value.mobility_min_rel_pct} step={0.5} min={0} label="Minimum mobility difference" width={80} onCommit={set("mobility_min_rel_pct")} /></Setting>
        <Setting label="Mobility precision (rel. SD)">
          <NumberInput value={value.mobility_precision_rel} step={0.001} min={0.0001} max={0.49} label="Mobility precision" width={90} onCommit={(v) => onChange({ ...value, mobility_precision_rel: v })} />
          {value.mobility_precision_rel == null && <span className="badge badge-warn">placeholder 0.005</span>}
        </Setting>
        <Setting label="MS/MS similarity ≥"><NumberInput value={value.similarity_threshold} step={0.05} min={0} max={1} label="Similarity threshold" width={80} onCommit={set("similarity_threshold")} /></Setting>
        <Setting label="Feature S/N ≥"><NumberInput value={value.min_snr} step={1} min={1} label="Minimum S/N" width={80} onCommit={set("min_snr")} /></Setting>
      </div>
    </div>
  );
}

function Setting({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="row" style={{ gap: 6 }}>{children}</div>
    </div>
  );
}

function RunResults({ run, opened, onOpen }: { run: DiscoveryRun; opened: number | null; onOpen: (id: number | null) => void }) {
  const [classes, setClasses] = useState<string[]>([]);
  const [category, setCategory] = useState<string>("");
  const [showExcluded, setShowExcluded] = useState(false);
  const [search, setSearch] = useState("");
  const counts = run.meta.counts;
  const notify = useStore((s) => s.notify);

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return run.candidates
      .filter((c) => showExcluded || c.status !== "excluded")
      .filter((c) => !classes.length || classes.includes(CLASS_SHORT[c.classification]))
      .filter((c) => !category || c.category === category)
      .filter((c) => !q || (c.mod_key ?? "").toLowerCase().includes(q) || c.neutral_mass.toFixed(3).includes(q))
      .sort((a, b) => rank(a) - rank(b) || b.neutral_mass - a.neutral_mass);
  }, [run, classes, category, showExcluded, search]);

  const high = counts.by_category["High-confidence putative candidate"] ?? 0;
  const toggleClass = (k: string) => setClasses((c) => (c.includes(k) ? c.filter((x) => x !== k) : [...c, k]));
  const report = (fmt: string) =>
    downloadGet(`/discovery/runs/${run.id}/report?format=${fmt}`).catch((e) => notify("err", String(e)));

  return (
    <>
      <div className="tiles">
        <Tile label="Candidates" value={counts.candidates} sub={`${counts.excluded} excluded (kept with reasons)`} />
        <Tile label="High-confidence putative" value={high} sub="MS/MS-validated, both species" />
        {Object.entries(counts.by_class).map(([k, v]) => (
          <Tile key={k} label={k} value={v} />
        ))}
      </div>
      {run.meta.notes.length > 0 && (
        <Banner kind="info" title="Run notes">
          {run.meta.notes.join(" · ")}
        </Banner>
      )}
      <section className="card" aria-labelledby="cand-h">
        <div className="card-head">
          <h2 id="cand-h">Candidates</h2>
          <div className="row">
            <div className="segmented" role="group" aria-label="Filter by class">
              {["A", "B", "C", "D"].map((k) => (
                <button key={k} type="button" aria-pressed={classes.includes(k)} onClick={() => toggleClass(k)} title={Object.keys(CLASS_SHORT).find((x) => CLASS_SHORT[x] === k)}>
                  {k}
                </button>
              ))}
            </div>
            <select className="select input-sm" value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Filter by category">
              <option value="">All categories</option>
              {Object.keys(CATEGORY_BADGE).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <label className="check" style={{ fontSize: 13 }}>
              <input type="checkbox" checked={showExcluded} onChange={(e) => setShowExcluded(e.target.checked)} /> Show excluded
            </label>
            <input className="input input-sm" placeholder="Sequence or mass" value={search} onChange={(e) => setSearch(e.target.value)} aria-label="Search candidates" style={{ width: 160 }} />
            <button className="btn btn-sm" onClick={() => void report("csv")}>
              <Download size={13} aria-hidden /> CSV
            </button>
            <button className="btn btn-sm" onClick={() => void report("html")}>
              <FileText size={13} aria-hidden /> Report
            </button>
          </div>
        </div>
        {rows.length === 0 ? (
          <p className="muted" style={{ padding: 16 }}>No candidates match the filters.</p>
        ) : (
          <div className="table-wrap" style={{ maxHeight: 460 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Category</th>
                  <th>Sequence</th>
                  <th className="r">z</th>
                  <th className="r">Neutral mass</th>
                  <th className="r">ΔRT (min)</th>
                  <th className="r">Δ mobility</th>
                  <th>MS/MS</th>
                  <th className="r">Flags</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((c) => (
                  <tr
                    key={c.candidate_id}
                    aria-selected={opened === c.candidate_id}
                    onClick={() => onOpen(c.candidate_id)}
                    onKeyDown={(e) => e.key === "Enter" && onOpen(c.candidate_id)}
                    tabIndex={0}
                    style={{ cursor: "pointer", opacity: c.status === "excluded" ? 0.65 : 1 }}
                  >
                    <td>
                      <span className="badge" title={c.classification}>
                        {CLASS_SHORT[c.classification]} · {c.classification.split(": ")[1]}
                      </span>
                    </td>
                    <td>
                      {c.status === "excluded" ? (
                        <span className="badge badge-err">Excluded</span>
                      ) : (
                        <span className={`badge ${CATEGORY_BADGE[c.category] ?? ""}`}>{c.category.replace(" candidate", "")}</span>
                      )}
                    </td>
                    <td className="mono" style={{ fontSize: 12.5 }}>{prettyKey(c.mod_key)}</td>
                    <td className="r num">{c.charge}+</td>
                    <td className="r num">{c.neutral_mass.toFixed(4)}</td>
                    <td className="r num">{c.species_b ? (c.delta_rt ?? 0).toFixed(2) : "–"}</td>
                    <td className="r num">{c.mobility_rel_diff_pct != null ? `${c.mobility_rel_diff_pct.toFixed(2)}%` : "–"}</td>
                    <td style={{ fontSize: 12.5 }}>
                      {c.msms_status ?? "–"}
                      {c.msms_cosine != null && <span className="faint num"> · {c.msms_cosine.toFixed(2)}</span>}
                    </td>
                    <td className="r">
                      {(c.flags.length > 0 || c.exclusion_reasons.length > 0) && (
                        <span className="row" style={{ gap: 3, justifyContent: "flex-end", flexWrap: "nowrap" }} title={[...c.exclusion_reasons, ...c.flags].join("\n")}>
                          <AlertTriangle size={13} aria-hidden style={{ color: "var(--accent)" }} />
                          <span className="num">{c.flags.length + c.exclusion_reasons.length}</span>
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {opened != null && <CandidateViewer runId={run.id} candidateId={opened} onClose={() => onOpen(null)} />}
    </>
  );
}

function rank(c: DiscoveryCandidate): number {
  if (c.status === "excluded") return 9;
  return ["High-confidence putative candidate", "Sequence-supported candidate", "Mass-matched candidate", "Unresolved candidate"].indexOf(c.category);
}

function Tile({ label, value, sub }: { label: string; value: number; sub?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

