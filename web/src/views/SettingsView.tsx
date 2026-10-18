import { FileDown, FileUp, RotateCcw } from "lucide-react";
import { useRef } from "react";

import { saveBlob } from "../api/client";
import { NumberInput, Segmented, StatusIcon } from "../components/ui";
import { currentSession, DEFAULT_QUERY, DEFAULT_STATS, type Session, useStore } from "../store";

export function SettingsView() {
  const query = useStore((s) => s.query);
  const stats = useStore((s) => s.stats);
  const theme = useStore((s) => s.theme);
  const caps = useStore((s) => s.caps);
  const { setQuery, setStats, setTheme, loadSession, notify } = useStore.getState();
  const fileRef = useRef<HTMLInputElement>(null);

  const save = () => {
    const blob = new Blob([JSON.stringify(currentSession(), null, 1)], { type: "application/json" });
    saveBlob(blob, `session-${new Date().toISOString().slice(0, 10)}.lctims.json`);
  };
  const load = async (file: File) => {
    try {
      const s = JSON.parse(await file.text()) as Session;
      if (s.version !== 1 || !Array.isArray(s.targets) || !Array.isArray(s.windows)) throw new Error("not an LCTIMSpy session file");
      loadSession(s);
      notify("ok", `Loaded session ${file.name}`);
    } catch (e) {
      notify("err", `Cannot load ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Settings</h1>
          <p className="muted">Analysis parameters apply to every chart and result.</p>
        </div>
        <div className="row">
          <button className="btn" onClick={save}>
            <FileDown size={14} aria-hidden /> Save session
          </button>
          <button className="btn" onClick={() => fileRef.current?.click()}>
            <FileUp size={14} aria-hidden /> Load session
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".json,application/json"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void load(f);
              e.target.value = "";
            }}
          />
        </div>
      </div>

      <div className="grid-2">
        <section className="card" aria-labelledby="q-h">
          <div className="card-head">
            <h2 id="q-h">Signal extraction</h2>
            <button className="btn btn-ghost btn-sm" onClick={() => setQuery(DEFAULT_QUERY)}>
              <RotateCcw size={13} aria-hidden /> Defaults
            </button>
          </div>
          <div className="card-body stack" style={{ gap: 16 }}>
            <Setting label="m/z tolerance" hint="One tolerance for chromatograms and mobilograms.">
              <NumberInput
                value={query.tolerance.value}
                step={query.tolerance.unit === "Da" ? 0.005 : 1}
                min={0.000001}
                label="m/z tolerance"
                width={96}
                onCommit={(value) => setQuery({ tolerance: { ...query.tolerance, value } })}
              />
              <Segmented
                label="Tolerance unit"
                value={query.tolerance.unit ?? "Da"}
                options={[
                  { value: "Da", label: "Da" },
                  { value: "ppm", label: "ppm" },
                ]}
                onChange={(unit) => setQuery({ tolerance: { value: unit === "ppm" ? 20 : 0.02, unit } })}
              />
            </Setting>
            <Setting label="Mobilogram smoothing" hint="Gaussian, in mobility bins. Off reproduces the unsmoothed legacy output.">
              <label className="check">
                <input
                  type="checkbox"
                  checked={query.mobilogram_sigma != null}
                  onChange={(e) => setQuery({ mobilogram_sigma: e.target.checked ? 1 : null })}
                />
                Smooth
              </label>
              {query.mobilogram_sigma != null && (
                <NumberInput value={query.mobilogram_sigma} step={0.5} min={0.1} label="Sigma" width={80} onCommit={(v) => setQuery({ mobilogram_sigma: v })} />
              )}
            </Setting>
            <Setting label="Baseline correction" hint="Subtract each mobilogram's minimum, as Bruker DataAnalysis does.">
              <label className="check">
                <input type="checkbox" checked={query.baseline_correction} onChange={(e) => setQuery({ baseline_correction: e.target.checked })} />
                Subtract baseline
              </label>
            </Setting>
            <Setting
              label="Intensity threshold"
              hint="A target is excluded for a sample unless every window reaches this maximum intensity."
            >
              <label className="check">
                <input
                  type="checkbox"
                  checked={query.intensity_threshold != null}
                  onChange={(e) => setQuery({ intensity_threshold: e.target.checked ? 100 : null })}
                />
                Enabled
              </label>
              {query.intensity_threshold != null && (
                <>
                  <NumberInput value={query.intensity_threshold} step={10} min={0} label="Threshold" width={96} onCommit={(v) => setQuery({ intensity_threshold: v })} />
                  <Segmented
                    label="Compare threshold with"
                    value={query.threshold_on}
                    options={[
                      { value: "corrected", label: "After baseline" },
                      { value: "raw", label: "Before baseline" },
                    ]}
                    onChange={(threshold_on) => setQuery({ threshold_on })}
                  />
                </>
              )}
            </Setting>
            {query.intensity_threshold != null && query.threshold_on === "corrected" && query.baseline_correction && (
              <p className="faint" style={{ fontSize: 12 }}>
                "After baseline" matches the original notebook, but makes the threshold's meaning depend on baseline correction.
              </p>
            )}
          </div>
        </section>

        <section className="card" aria-labelledby="s-h">
          <div className="card-head">
            <h2 id="s-h">Statistics</h2>
            <button className="btn btn-ghost btn-sm" onClick={() => setStats(DEFAULT_STATS)}>
              <RotateCcw size={13} aria-hidden /> Defaults
            </button>
          </div>
          <div className="card-body stack" style={{ gap: 16 }}>
            <Setting label="Metric">
              <Segmented
                label="Metric"
                value={stats.metric}
                options={[
                  { value: "mobility_difference", label: "Mean mobility" },
                  { value: "peak_mobility_difference", label: "Peak mobility" },
                ]}
                onChange={(metric) => setStats({ metric })}
              />
            </Setting>
            <Setting label="Test">
              <Segmented
                label="t-test variant"
                value={stats.equal_var ? "student" : "welch"}
                options={[
                  { value: "student", label: "Student's t" },
                  { value: "welch", label: "Welch's t" },
                ]}
                onChange={(v) => setStats({ equal_var: v === "student" })}
              />
            </Setting>
            <Setting label="Multiple-testing correction">
              <select
                className="select"
                value={stats.correction}
                onChange={(e) => setStats({ correction: e.target.value as typeof stats.correction })}
                aria-label="Multiple-testing correction"
              >
                <option value="bonferroni">Bonferroni</option>
                <option value="holm">Holm</option>
                <option value="tukey">Tukey HSD</option>
                <option value="none">None</option>
              </select>
            </Setting>
            <Setting label="Significance level α">
              <NumberInput value={stats.alpha} step={0.01} min={0.0001} max={0.5} label="Alpha" width={80} onCommit={(alpha) => setStats({ alpha })} />
            </Setting>
            <Setting label="Minimum samples per target" hint="Targets measured in fewer samples are left out of the tests.">
              <NumberInput value={stats.min_samples} step={1} min={2} label="Minimum samples" width={80} onCommit={(v) => setStats({ min_samples: Math.round(v) })} />
            </Setting>
          </div>
        </section>
      </div>

      <div className="grid-2">
        <section className="card" aria-labelledby="a-h">
          <div className="card-head">
            <h2 id="a-h">Appearance</h2>
          </div>
          <div className="card-body">
            <Setting label="Theme">
              <Segmented
                label="Theme"
                value={theme}
                options={[
                  { value: "system", label: "System" },
                  { value: "light", label: "Light" },
                  { value: "dark", label: "Dark" },
                ]}
                onChange={setTheme}
              />
            </Setting>
          </div>
        </section>
        <section className="card" aria-labelledby="sys-h">
          <div className="card-head">
            <h2 id="sys-h">System</h2>
          </div>
          <div className="card-body stack" style={{ gap: 8, fontSize: 13 }}>
            {caps ? (
              <>
                <div className="row">
                  <StatusIcon ok={caps.sdk.available} />
                  <span>
                    Bruker TDF-SDK: {caps.sdk.available ? `available (pyTDFSDK ${caps.sdk.version ?? "?"})` : "unavailable"}
                  </span>
                </div>
                {!caps.sdk.available && <p className="muted">{caps.sdk.error} {caps.sdk.remediation}</p>}
                <div className="muted">
                  LCTIMSpy {caps.version} · {caps.platform}
                </div>
                <div className="muted">
                  Cache: <span className="mono">{caps.cache_dir}</span>
                </div>
              </>
            ) : (
              <span className="muted">Connecting…</span>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function Setting({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="row">{children}</div>
      {hint && <small>{hint}</small>}
    </div>
  );
}
