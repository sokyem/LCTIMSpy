import { CheckCircle2, ChevronDown, ChevronUp, CircleSlash, Loader2, Square, XCircle } from "lucide-react";
import { useState } from "react";

import { api, type Job } from "../api/client";
import { fmtBytes, fmtEta } from "../lib/format";
import { useStore } from "../store";

const ACTIVE = new Set(["queued", "running", "cancelling"]);

function stageLabel(job: Job): string {
  const p = job.progress as { stage?: string } | null;
  if (job.state === "queued") return "Queued";
  if (job.state === "cancelling") return "Cancelling…";
  if (p?.stage === "finalize") return "Sorting by m/z";
  if (p?.stage === "copy") return "Copying";
  if (p?.stage === "cached") return "Already cached";
  return job.kind === "copy-local" ? "Copying" : "Extracting";
}

export function JobBar() {
  const jobs = useStore((s) => s.jobs);
  const datasets = useStore((s) => s.datasets);
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState<Record<string, Job>>({});

  const list = Object.values(jobs).sort((a, b) => b.created - a.created);
  const active = list.filter((j) => ACTIVE.has(j.state));
  const recent = list.filter((j) => !ACTIVE.has(j.state)).slice(0, 8);
  const failed = recent.filter((j) => j.state === "failed").length;
  const shown = [...active, ...recent];

  const loadDetail = async (job: Job) => {
    if (details[job.id]) return;
    const full = await api.get<Job>(`/jobs/${job.id}`);
    setDetails((d) => ({ ...d, [job.id]: full }));
  };

  const head = active.length
    ? `${active.length} job${active.length > 1 ? "s" : ""} running`
    : recent.length
      ? "No jobs running"
      : "No jobs yet";

  return (
    <footer className="jobbar" aria-label="Background jobs">
      <div className="jobbar-head">
        {active.length ? <Loader2 size={16} className="spin" aria-hidden /> : <CheckCircle2 size={16} aria-hidden className="faint" />}
        <strong style={{ fontSize: 13 }}>{head}</strong>
        {active[0] && <JobProgress job={active[0]} compact />}
        {failed > 0 && <span className="badge badge-err">{failed} failed</span>}
        <span className="spacer" />
        {shown.length > 0 && (
          <button className="btn btn-ghost btn-sm" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />} {open ? "Hide" : "Show"} jobs
          </button>
        )}
      </div>
      {open && (
        <div className="jobs-list">
          {shown.map((job) => {
            const name = (job.dataset_id && datasets[job.dataset_id]?.name) || job.dataset_id || job.kind;
            const detail = details[job.id];
            return (
              <div className="job" key={job.id}>
                <JobIcon state={job.state} />
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis" }}>{name}</div>
                  <div className="faint" style={{ fontSize: 12 }}>
                    {job.kind === "copy-local" ? "Copy to local disk" : "Extraction"} · {stateText(job)}
                  </div>
                </div>
                {ACTIVE.has(job.state) ? <JobProgress job={job} /> : <span className="faint">{resultText(job)}</span>}
                <div className="row" style={{ justifyContent: "flex-end" }}>
                  {ACTIVE.has(job.state) && job.state !== "cancelling" && (
                    <button className="btn btn-sm" onClick={() => void api.del(`/jobs/${job.id}`)}>
                      <Square size={12} aria-hidden /> Cancel
                    </button>
                  )}
                  {job.state === "failed" && !detail && (
                    <button className="btn btn-sm" onClick={() => void loadDetail(job)}>
                      Details
                    </button>
                  )}
                </div>
                {job.state === "failed" && job.error && (
                  <div className="job-error">
                    {String((job.error as { message?: string }).message ?? "")}
                    {(job.error as { remediation?: string }).remediation &&
                      `\nFix: ${(job.error as { remediation?: string }).remediation}`}
                    {detail?.log_tail?.length ? `\n\n${detail.log_tail.slice(-30).join("\n")}` : ""}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </footer>
  );
}

function JobProgress({ job, compact = false }: { job: Job; compact?: boolean }) {
  const p = (job.progress ?? {}) as { done?: number; total?: number; eta_s?: number | null; rate?: number | null };
  const frac = p.total ? Math.min(1, (p.done ?? 0) / p.total) : 0;
  return (
    <div className="row" style={{ gap: 8, flex: compact ? "0 1 360px" : 1, minWidth: 0, flexWrap: "nowrap" }}>
      <div
        className="progress"
        style={{ flex: 1 }}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(frac * 100)}
        aria-label={`${stageLabel(job)} progress`}
      >
        <div style={{ width: `${frac * 100}%` }} />
      </div>
      <span className="num faint" style={{ whiteSpace: "nowrap", fontSize: 12 }}>
        {stageLabel(job)} {Math.round(frac * 100)}%{p.eta_s ? ` · ${fmtEta(p.eta_s)} left` : ""}
      </span>
    </div>
  );
}

function JobIcon({ state }: { state: Job["state"] }) {
  if (ACTIVE.has(state)) return <Loader2 size={16} className="spin" aria-label={state} />;
  if (state === "done") return <CheckCircle2 size={16} aria-label="done" style={{ color: "var(--success)" }} />;
  if (state === "failed") return <XCircle size={16} aria-label="failed" style={{ color: "var(--danger)" }} />;
  return <CircleSlash size={16} aria-label="cancelled" className="faint" />;
}

function stateText(job: Job): string {
  return { queued: "Queued", running: "Running", cancelling: "Cancelling", done: "Done", failed: "Failed", cancelled: "Cancelled" }[job.state];
}

function resultText(job: Job): string {
  const r = job.result as { rows?: number; bytes?: number; skipped?: boolean } | null;
  if (job.state !== "done" || !r) return "";
  if (job.kind === "copy-local") return r.skipped ? "Local copy already up to date" : `Copied ${fmtBytes(r.bytes)}`;
  return `${(r.rows ?? 0).toLocaleString()} points · ${fmtBytes(r.bytes)}`;
}
