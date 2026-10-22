import { Copy, Database, FolderPlus, HardDriveDownload, Loader2, Network, Play, RotateCcw, Trash2 } from "lucide-react";
import { useState } from "react";

import { api, ApiError, type Dataset, type Job } from "../api/client";
import { FsBrowser } from "../components/FsBrowser";
import { Banner, Empty } from "../components/ui";
import { fmtBytes, fmtRt } from "../lib/format";
import { useStore } from "../store";

const ACTIVE = new Set(["queued", "running", "cancelling"]);

export function DatasetsView() {
  const datasets = useStore((s) => s.datasets);
  const jobs = useStore((s) => s.jobs);
  const selected = useStore((s) => s.selected);
  const caps = useStore((s) => s.caps);
  const toggleSelected = useStore((s) => s.toggleSelected);
  const notify = useStore((s) => s.notify);
  const setView = useStore((s) => s.setView);
  const [adding, setAdding] = useState(false);

  const list = Object.values(datasets).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const activeJob = (id: string, kind: string): Job | undefined =>
    Object.values(jobs).find((j) => j.dataset_id === id && j.kind === kind && ACTIVE.has(j.state));
  const sdkOk = caps?.sdk.available ?? true;
  const ready = list.filter((d) => selected.includes(d.id) && d.cache.status === "cached").length;
  const pending = list.filter((d) => selected.includes(d.id) && d.cache.status !== "cached");

  const add = async (paths: string[]) => {
    const errors: string[] = [];
    for (const path of paths) {
      try {
        const d = await api.post<Dataset>("/datasets", { path });
        useStore.getState().upsertDataset(d);
        toggleSelected(d.id, true);
      } catch (e) {
        errors.push(`${path}: ${e instanceof ApiError ? e.message : String(e)}`);
      }
    }
    if (errors.length) throw new Error(errors.join("\n"));
    notify("ok", `Added ${paths.length} dataset${paths.length > 1 ? "s" : ""}`);
  };

  const run = async (path: string, body: object, label: string) => {
    try {
      const job = await api.post<Job>(path, body);
      useStore.getState().upsertJob(job);
    } catch (e) {
      notify("err", `${label}: ${e instanceof ApiError ? e.message : String(e)}`);
    }
  };
  const extract = (d: Dataset, force = false) => run("/jobs/extract", { dataset_id: d.id, force }, `Extract ${d.name}`);
  const copyLocal = (d: Dataset) => run("/jobs/copy-local", { dataset_id: d.id }, `Copy ${d.name}`);
  const remove = async (d: Dataset) => {
    if (!confirm(`Remove ${d.name} from the list? The data and its cache are kept.`)) return;
    await api.del(`/datasets/${d.id}`);
    useStore.getState().removeDataset(d.id);
  };
  const setGroup = async (d: Dataset, group: string) => {
    const g = group.trim() || null;
    if (g === d.group) return;
    useStore.getState().upsertDataset(await api.patch<Dataset>(`/datasets/${d.id}`, { group: g }));
  };

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Datasets</h1>
          <p className="muted">Add timsTOF .d folders or mzML files, extract them once, and select the replicates to analyse.</p>
        </div>
        <div className="row">
          {pending.length > 0 && sdkOk && (
            <button className="btn" onClick={() => pending.forEach((d) => !activeJob(d.id, "extract") && void extract(d))}>
              <Play size={14} aria-hidden /> Extract {pending.length} selected
            </button>
          )}
          <button className="btn btn-primary" onClick={() => setAdding(true)}>
            <FolderPlus size={16} aria-hidden /> Add datasets
          </button>
        </div>
      </div>

      {caps && !caps.sdk.available && (
        <Banner kind="warn" title="Extraction is unavailable on this machine">
          {caps.sdk.error} {caps.sdk.remediation && <>Fix: <span className="mono">{caps.sdk.remediation}</span>. </>}
          Datasets that are already extracted can still be analysed.
        </Banner>
      )}

      <div className="card">
        {list.length === 0 ? (
          <Empty icon={<Database size={32} aria-hidden />} title="No datasets yet">
            <p>Add the .d folders (or mzML files) of your LC-TIMS runs to get started.</p>
            <button className="btn btn-primary" onClick={() => setAdding(true)} style={{ marginTop: 8 }}>
              <FolderPlus size={16} aria-hidden /> Add datasets
            </button>
          </Empty>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th style={{ width: 36 }}>
                    <span className="sr-only">Selected for analysis</span>
                  </th>
                  <th>Dataset</th>
                  <th>Group</th>
                  <th className="r">RT range (min)</th>
                  <th className="r">MS1 frames</th>
                  <th>Extraction</th>
                  <th className="r">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {list.map((d) => {
                  const job = activeJob(d.id, "extract");
                  const copyJob = activeJob(d.id, "copy-local");
                  const isSel = selected.includes(d.id);
                  return (
                    <tr key={d.id} aria-selected={isSel}>
                      <td>
                        <input
                          type="checkbox"
                          className="check"
                          checked={isSel}
                          onChange={(e) => toggleSelected(d.id, e.target.checked)}
                          aria-label={`Analyse ${d.name}`}
                          style={{ width: 16, height: 16, accentColor: "var(--primary)" }}
                        />
                      </td>
                      <td style={{ maxWidth: 380 }}>
                        <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                          <strong>{d.name}</strong>
                          {d.is_network && !d.local_path && (
                            <span className="badge badge-warn" title="On a network drive">
                              <Network size={12} aria-hidden /> Network
                            </span>
                          )}
                          {d.local_path && (
                            <span className="badge badge-info" title={d.local_path}>
                              <HardDriveDownload size={12} aria-hidden /> Local copy
                            </span>
                          )}
                          {!d.available && <span className="badge badge-err">Folder missing</span>}
                        </div>
                        <div className="faint mono" style={{ fontSize: 11.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={d.path}>
                          {d.path}
                        </div>
                      </td>
                      <td>
                        <input
                          className="input input-sm"
                          style={{ width: 110 }}
                          defaultValue={d.group ?? ""}
                          placeholder="—"
                          aria-label={`Group for ${d.name}`}
                          onBlur={(e) => void setGroup(d, e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
                        />
                      </td>
                      <td className="r num">
                        {fmtRt(d.meta.rt_min as number)}–{fmtRt(d.meta.rt_max as number)}
                      </td>
                      <td className="r num">{(d.meta.n_ms1_frames as number).toLocaleString()}</td>
                      <td>
                        <CacheCell d={d} job={job} />
                      </td>
                      <td className="r">
                        <div className="row" style={{ justifyContent: "flex-end", flexWrap: "nowrap", gap: 4 }}>
                          {d.cache.status !== "cached" && !job && (
                            <button className="btn btn-sm" disabled={!sdkOk || !d.available} onClick={() => void extract(d)}>
                              <Play size={13} aria-hidden /> Extract
                            </button>
                          )}
                          {d.cache.status === "cached" && !job && (
                            <button
                              className="btn btn-ghost btn-icon btn-sm"
                              disabled={!sdkOk || !d.available}
                              onClick={() => confirm(`Re-extract ${d.name}, ignoring the cache?`) && void extract(d, true)}
                              aria-label={`Re-extract ${d.name}`}
                              title="Re-extract (ignore cache)"
                            >
                              <RotateCcw size={14} />
                            </button>
                          )}
                          {d.is_network && !d.local_path && (
                            <button
                              className="btn btn-ghost btn-icon btn-sm"
                              disabled={!!copyJob}
                              onClick={() => void copyLocal(d)}
                              aria-label={`Copy ${d.name} to local disk`}
                              title="Copy to local disk (faster extraction)"
                            >
                              {copyJob ? <Loader2 size={14} className="spin" /> : <Copy size={14} />}
                            </button>
                          )}
                          <button
                            className="btn btn-ghost btn-icon btn-sm btn-danger"
                            onClick={() => void remove(d)}
                            aria-label={`Remove ${d.name}`}
                            title="Remove from list"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {list.length > 0 && (
        <div className="row">
          <span className="muted">
            {selected.length} selected · {ready} ready to analyse
            {selected.length > ready && ` · ${selected.length - ready} still need extraction`}
          </span>
          <span className="spacer" />
          <button className="btn btn-primary" disabled={ready === 0} onClick={() => setView("targets")}>
            Next: choose targets
          </button>
        </div>
      )}

      {adding && <FsBrowser onClose={() => setAdding(false)} onAdd={add} />}
    </div>
  );
}

function CacheCell({ d, job }: { d: Dataset; job?: Job }) {
  if (job) {
    const p = (job.progress ?? {}) as { done?: number; total?: number; stage?: string };
    const pct = p.total ? Math.round(((p.done ?? 0) / p.total) * 100) : 0;
    return (
      <span className="badge badge-info">
        <Loader2 size={12} className="spin" aria-hidden />
        {job.state === "queued" ? "Queued" : p.stage === "finalize" ? `Sorting ${pct}%` : `Extracting ${pct}%`}
      </span>
    );
  }
  if (d.cache.status === "cached") {
    return (
      <span className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
        <span className="badge badge-ok">Extracted</span>
        <span className="faint num" style={{ fontSize: 12 }}>
          {fmtBytes(d.cache.bytes)}
        </span>
        {(d.cache.frames_failed ?? 0) > 0 && (
          <span className="badge badge-warn" title="Some frames could not be read">
            {d.cache.frames_failed} frames failed
          </span>
        )}
      </span>
    );
  }
  return <span className="badge">Not extracted</span>;
}
