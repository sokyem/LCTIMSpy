import { Calculator, FlaskConical, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { api, ApiError, type ApiTarget } from "../api/client";
import { Banner, Empty, NumberInput } from "../components/ui";
import { fmtMz } from "../lib/format";
import { MAX_SERIES, seriesColor } from "../lib/palette";
import { useDarkMode } from "../lib/theme";
import { enabledTargets, useStore } from "../store";

interface Fragments {
  targets: ApiTarget[];
  precursors: Record<string, number>;
}

export function TargetsView() {
  const dark = useDarkMode();
  const peptide = useStore((s) => s.peptide);
  const setPeptide = useStore((s) => s.setPeptide);
  const targets = useStore((s) => s.targets);
  const addTargets = useStore((s) => s.addTargets);
  const setTargetEnabled = useStore((s) => s.setTargetEnabled);
  const updateTarget = useStore((s) => s.updateTarget);
  const removeTarget = useStore((s) => s.removeTarget);
  const clearTargets = useStore((s) => s.clearTargets);
  const notify = useStore((s) => s.notify);
  const setView = useStore((s) => s.setView);

  const [fragments, setFragments] = useState<Fragments | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [manual, setManual] = useState("");
  const [filter, setFilter] = useState("");

  const compute = async () => {
    setBusy(true);
    setError(null);
    try {
      setFragments(await api.post<Fragments>("/peptides/fragments", {
        peptide: peptide.sequence.trim(),
        charges: peptide.charges,
      }));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setFragments(null);
    } finally {
      setBusy(false);
    }
  };

  const add = (ts: ApiTarget[]) => {
    const warning = addTargets(ts.map((t) => ({ mz: t.mz, label: t.label ?? null })));
    if (warning) notify("err", warning);
  };

  const addManual = () => {
    const values = manual.split(/[\s,;]+/).map(Number).filter((n) => Number.isFinite(n) && n > 0);
    if (!values.length) return;
    add(values.map((mz) => ({ mz, label: null })));
    setManual("");
  };

  const toggle = (i: number, on: boolean) => {
    const warning = setTargetEnabled(i, on);
    if (warning) notify("err", warning);
  };

  const inList = (mz: number) => targets.some((t) => Math.abs(t.mz - mz) < 1e-6);
  const shownFragments = (fragments?.targets ?? []).filter(
    (f) => !filter || (f.label ?? "").toLowerCase().includes(filter.toLowerCase()),
  );
  const nEnabled = enabledTargets(targets).length;

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Targets</h1>
          <p className="muted">Fragment ions to follow. Each plotted target keeps one colour across every chart.</p>
        </div>
        <button className="btn btn-primary" disabled={nEnabled === 0} onClick={() => setView("windows")}>
          Next: set RT windows
        </button>
      </div>

      <div className="grid-2">
        <section className="card" aria-labelledby="pep-h">
          <div className="card-head">
            <h2 id="pep-h">From a peptide</h2>
          </div>
          <div className="card-body stack">
            <div className="row" style={{ alignItems: "flex-end" }}>
              <label className="field" style={{ flex: "1 1 220px" }}>
                <span>Sequence</span>
                <input
                  className="input mono"
                  value={peptide.sequence}
                  placeholder="SPQLEDEAKELQ(-0.98)"
                  spellCheck={false}
                  onChange={(e) => setPeptide({ sequence: e.target.value.toUpperCase() })}
                  onKeyDown={(e) => e.key === "Enter" && peptide.sequence.trim() && void compute()}
                />
              </label>
              <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
                <span>Charges</span>
                <div className="row" style={{ gap: 10 }}>
                  {[1, 2, 3].map((z) => (
                    <label className="check" key={z}>
                      <input
                        type="checkbox"
                        checked={peptide.charges.includes(z)}
                        onChange={(e) =>
                          setPeptide({
                            charges: e.target.checked
                              ? [...peptide.charges, z].sort()
                              : peptide.charges.filter((c) => c !== z),
                          })
                        }
                      />
                      {z}+
                    </label>
                  ))}
                </div>
              </fieldset>
              <button
                className="btn btn-primary"
                disabled={!peptide.sequence.trim() || !peptide.charges.length || busy}
                onClick={() => void compute()}
              >
                <Calculator size={14} aria-hidden /> Calculate
              </button>
            </div>
            <small className="faint">End the sequence with (-0.98) for C-terminal amidation.</small>
            {error && <Banner kind="err" title="Cannot compute fragments">{error}</Banner>}
            {fragments && (
              <>
                <div className="row">
                  {Object.entries(fragments.precursors).map(([k, v]) => (
                    <span key={k} className="badge">
                      {k} <span className="num">{fmtMz(v)}</span>
                    </span>
                  ))}
                  <span className="spacer" />
                  <input
                    className="input input-sm"
                    placeholder="Filter, e.g. y or (2+)"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    aria-label="Filter fragments"
                    style={{ width: 160 }}
                  />
                  <button className="btn btn-sm" onClick={() => add(shownFragments.filter((f) => !inList(f.mz)))}>
                    Add {filter ? "shown" : "all"}
                  </button>
                </div>
                <div className="table-wrap" style={{ maxHeight: 360, border: "1px solid var(--border)", borderRadius: 6 }}>
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Ion</th>
                        <th className="r">m/z</th>
                        <th className="r">
                          <span className="sr-only">Add</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {shownFragments.map((f) => (
                        <tr key={f.mz}>
                          <td>{f.label}</td>
                          <td className="r num">{fmtMz(f.mz)}</td>
                          <td className="r">
                            <button
                              className="btn btn-sm"
                              disabled={inList(f.mz)}
                              onClick={() => add([f])}
                              aria-label={`Add ${f.label}`}
                            >
                              {inList(f.mz) ? "Added" : <><Plus size={13} aria-hidden /> Add</>}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            <div className="row" style={{ alignItems: "flex-end", borderTop: "1px solid var(--border)", paddingTop: 12 }}>
              <label className="field" style={{ flex: 1 }}>
                <span>Or enter m/z values</span>
                <input
                  className="input mono"
                  value={manual}
                  placeholder="500.25, 700.40"
                  onChange={(e) => setManual(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && addManual()}
                />
              </label>
              <button className="btn" onClick={addManual} disabled={!manual.trim()}>
                <Plus size={14} aria-hidden /> Add
              </button>
            </div>
          </div>
        </section>

        <section className="card" aria-labelledby="tl-h">
          <div className="card-head">
            <h2 id="tl-h">Target list</h2>
            <div className="row">
              <span className="faint" style={{ fontSize: 12 }}>
                {nEnabled} of {MAX_SERIES} plotted
              </span>
              {targets.length > 0 && (
                <button className="btn btn-ghost btn-sm btn-danger" onClick={() => confirm("Remove all targets?") && clearTargets()}>
                  Clear
                </button>
              )}
            </div>
          </div>
          {targets.length === 0 ? (
            <Empty icon={<FlaskConical size={32} aria-hidden />} title="No targets yet">
              Calculate fragments from the peptide, or enter m/z values.
            </Empty>
          ) : (
            <div className="table-wrap" style={{ maxHeight: 560 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th style={{ width: 70 }}>Plot</th>
                    <th>Label</th>
                    <th className="r">m/z</th>
                    <th className="r">
                      <span className="sr-only">Remove</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {targets.map((t, i) => {
                    const color = seriesColor(t.slot, dark);
                    return (
                      <tr key={t.mz}>
                        <td>
                          <label className="check">
                            <input
                              type="checkbox"
                              checked={t.enabled}
                              onChange={(e) => toggle(i, e.target.checked)}
                              aria-label={`Plot ${t.label ?? fmtMz(t.mz)}`}
                            />
                            <span
                              className={`swatch ${color ? "" : "swatch-empty"}`}
                              style={color ? { background: color } : undefined}
                              aria-hidden
                            />
                          </label>
                        </td>
                        <td>
                          <input
                            className="input input-sm"
                            defaultValue={t.label ?? ""}
                            placeholder={`m/z ${fmtMz(t.mz)}`}
                            aria-label="Label"
                            onBlur={(e) => updateTarget(i, { label: e.target.value.trim() || null })}
                            style={{ width: "100%", minWidth: 100 }}
                          />
                        </td>
                        <td className="r">
                          <NumberInput
                            value={t.mz}
                            step={0.0001}
                            min={1}
                            label="m/z"
                            width={110}
                            onCommit={(mz) => updateTarget(i, { mz })}
                          />
                        </td>
                        <td className="r">
                          <button
                            className="btn btn-ghost btn-icon btn-sm"
                            onClick={() => removeTarget(i)}
                            aria-label={`Remove ${t.label ?? fmtMz(t.mz)}`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
