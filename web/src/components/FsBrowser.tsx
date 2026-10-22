import { ChevronRight, Database, FileText, Folder, HardDrive, Loader2, Network } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { api, ApiError, type FsEntry, type FsListing } from "../api/client";
import { Modal } from "./Modal";
import { Banner } from "./ui";

interface Props {
  onClose: () => void;
  onAdd: (paths: string[]) => Promise<void>;
  /** "datasets": several .d folders / mzML files. "file": one file with ``extensions``. */
  mode?: "datasets" | "file";
  extensions?: string[];
  title?: string;
}

/** Pick datasets (Bruker .d folders, mzML files) or one file. Paths can also be pasted. */
export function FsBrowser({ onClose, onAdd, mode = "datasets", extensions = [], title }: Props) {
  const [roots, setRoots] = useState<FsEntry[]>([]);
  const [listing, setListing] = useState<FsListing | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [typed, setTyped] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const single = mode === "file";

  const open = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      setListing(await api.get<FsListing>(`/fs/list?path=${encodeURIComponent(path)}`));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    api
      .get<FsEntry[]>("/fs/roots")
      .then((r) => {
        setRoots(r);
        if (r[0]) void open(r[0].path);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)));
  }, [open]);

  const toggle = (path: string) =>
    setPicked((p) => (single ? [path] : p.includes(path) ? p.filter((x) => x !== path) : [...p, path]));

  const typedPaths = typed.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const submit = async () => {
    const paths = single ? (typedPaths[0] ? [typedPaths[0]] : picked) : [...picked, ...typedPaths];
    if (!paths.length) return;
    setBusy(true);
    try {
      await onAdd(paths);
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const entries = listing?.entries ?? [];
  const matches = (e: FsEntry) => extensions.some((x) => e.name.toLowerCase().endsWith(x));
  const selectable = entries.filter((e) =>
    single ? e.kind !== "dir" && e.kind !== "d_folder" && matches(e) : e.kind === "d_folder" || e.kind === "mzml",
  );
  const dirs = entries.filter((e) => e.kind === "dir" || (single && e.kind === "d_folder"));
  const count = single ? (typedPaths.length || picked.length ? 1 : 0) : picked.length + typedPaths.length;
  const crumbs = listing ? crumbsFor(listing.path) : [];

  return (
    <Modal
      title={title ?? (single ? "Choose a file" : "Add datasets")}
      onClose={onClose}
      footer={
        <>
          <span
            className="muted"
            style={{ marginRight: "auto", alignSelf: "center", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
            title={single ? picked[0] ?? typedPaths[0] : undefined}
          >
            {single
              ? picked[0] ?? (typedPaths[0] || `Select a ${extensions.join(" / ")} file`)
              : count
                ? `${count} selected`
                : "Select .d folders or mzML files, or paste paths"}
          </span>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={!count || busy} onClick={submit}>
            {busy && <Loader2 size={14} className="spin" aria-hidden />}
            {single ? "Choose" : `Add ${count || ""}`}
          </button>
        </>
      }
    >
      <div className="card-body stack" style={{ gap: 10, borderBottom: "1px solid var(--border)" }}>
        <div className="row">
          {roots.map((r) => (
            <button key={r.path} className="btn btn-sm" onClick={() => void open(r.path)}>
              <HardDrive size={14} aria-hidden /> {r.name}
            </button>
          ))}
        </div>
        {listing && (
          <div className="crumbs" aria-label="Current folder">
            {crumbs.map((c, i) => (
              <span key={c.path} className="row" style={{ gap: 2 }}>
                {i > 0 && <ChevronRight size={12} aria-hidden className="faint" />}
                <button onClick={() => void open(c.path)}>{c.name}</button>
              </span>
            ))}
            {listing.is_network && (
              <span className="badge badge-warn" style={{ marginLeft: 8 }}>
                <Network size={12} aria-hidden /> Network drive
              </span>
            )}
          </div>
        )}
        {error && <Banner kind="err" title="Could not open">{error}</Banner>}
      </div>
      <div role="listbox" aria-multiselectable={!single} aria-label="Folder contents" aria-busy={loading}>
        {listing?.parent && (
          <button className="fs-row" onClick={() => void open(listing.parent!)}>
            <Folder size={16} aria-hidden className="faint" /> ..
          </button>
        )}
        {selectable.map((e) => (
          <button
            key={e.path}
            role="option"
            aria-selected={picked.includes(e.path)}
            className="fs-row"
            onClick={() => toggle(e.path)}
          >
            <input type={single ? "radio" : "checkbox"} readOnly checked={picked.includes(e.path)} tabIndex={-1} aria-hidden />
            {e.kind === "d_folder" || e.kind === "mzml" ? (
              <Database size={16} aria-hidden style={{ color: "var(--primary)" }} />
            ) : (
              <FileText size={16} aria-hidden style={{ color: "var(--primary)" }} />
            )}
            <strong>{e.name}</strong>
            <span className="faint">{e.kind === "d_folder" ? "timsTOF dataset" : e.kind === "mzml" ? "mzML" : "file"}</span>
          </button>
        ))}
        {dirs.map((e) => (
          <button key={e.path} className="fs-row" onClick={() => void open(e.path)}>
            <Folder size={16} aria-hidden className="faint" /> {e.name}
          </button>
        ))}
        {listing && !selectable.length && !dirs.length && (
          <p className="muted" style={{ padding: 16 }}>
            Nothing to pick here.
          </p>
        )}
      </div>
      <div className="card-body" style={{ borderTop: "1px solid var(--border)" }}>
        <label className="field">
          <span>Or paste {single ? "a path" : "paths (one per line)"}</span>
          <textarea
            className="input mono"
            rows={single ? 1 : 2}
            style={{ paddingTop: 6, resize: "vertical" }}
            placeholder={single ? "C:\\results\\db.psms.csv" : "X:\\peaks\\data\\Samuel\\20240707ISOASP\\RH_0325_cid_tims_Slot1-19_1_5956.d"}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
          />
        </label>
      </div>
    </Modal>
  );
}

function crumbsFor(path: string): { name: string; path: string }[] {
  const sep = path.includes("\\") ? "\\" : "/";
  const parts = path.split(/[\\/]/).filter(Boolean);
  const out: { name: string; path: string }[] = [];
  let acc = path.startsWith("/") ? "/" : "";
  for (const p of parts) {
    acc = acc && !acc.endsWith(sep) ? `${acc}${sep}${p}` : `${acc}${p}`;
    const full = /^[A-Za-z]:$/.test(acc) ? `${acc}${sep}` : acc;
    out.push({ name: p, path: full });
  }
  return out;
}
