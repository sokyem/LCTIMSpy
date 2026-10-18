import { ChevronRight, Database, Folder, HardDrive, Loader2, Network } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { api, ApiError, type FsEntry, type FsListing } from "../api/client";
import { Modal } from "./Modal";
import { Banner } from "./ui";

/** Pick one or more Bruker .d folders. Paths can also be pasted directly. */
export function FsBrowser({ onClose, onAdd }: { onClose: () => void; onAdd: (paths: string[]) => Promise<void> }) {
  const [roots, setRoots] = useState<FsEntry[]>([]);
  const [listing, setListing] = useState<FsListing | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [typed, setTyped] = useState("");
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setPicked((p) => (p.includes(path) ? p.filter((x) => x !== path) : [...p, path]));

  const submit = async () => {
    const paths = [...picked, ...typed.split(/\r?\n/).map((s) => s.trim()).filter(Boolean)];
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

  const dFolders = listing?.entries.filter((e) => e.kind === "d_folder") ?? [];
  const dirs = listing?.entries.filter((e) => e.kind === "dir") ?? [];
  const count = picked.length + typed.split(/\r?\n/).filter((s) => s.trim()).length;
  const crumbs = listing ? crumbsFor(listing.path) : [];

  return (
    <Modal
      title="Add datasets"
      onClose={onClose}
      footer={
        <>
          <span className="muted" style={{ marginRight: "auto", alignSelf: "center" }}>
            {count ? `${count} selected` : "Select .d folders or paste paths"}
          </span>
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={!count || busy} onClick={submit}>
            {busy && <Loader2 size={14} className="spin" aria-hidden />}
            Add {count || ""}
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
      <div role="listbox" aria-multiselectable="true" aria-label="Folder contents" aria-busy={loading}>
        {listing?.parent && (
          <button className="fs-row" onClick={() => void open(listing.parent!)}>
            <Folder size={16} aria-hidden className="faint" /> ..
          </button>
        )}
        {dFolders.map((e) => (
          <button
            key={e.path}
            role="option"
            aria-selected={picked.includes(e.path)}
            className="fs-row"
            onClick={() => toggle(e.path)}
          >
            <input type="checkbox" readOnly checked={picked.includes(e.path)} tabIndex={-1} aria-hidden />
            <Database size={16} aria-hidden style={{ color: "var(--primary)" }} />
            <strong>{e.name}</strong>
            <span className="faint">timsTOF dataset</span>
          </button>
        ))}
        {dirs.map((e) => (
          <button key={e.path} className="fs-row" onClick={() => void open(e.path)}>
            <Folder size={16} aria-hidden className="faint" /> {e.name}
          </button>
        ))}
        {listing && !dFolders.length && !dirs.length && (
          <p className="muted" style={{ padding: 16 }}>
            No folders here.
          </p>
        )}
      </div>
      <div className="card-body" style={{ borderTop: "1px solid var(--border)" }}>
        <label className="field">
          <span>Or paste paths (one per line)</span>
          <textarea
            className="input mono"
            rows={2}
            style={{ paddingTop: 6, resize: "vertical" }}
            placeholder="X:\peaks\data\Samuel\20240707ISOASP\RH_0325_cid_tims_Slot1-19_1_5956.d"
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
