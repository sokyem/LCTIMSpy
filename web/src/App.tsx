import { BarChart3, Crosshair, Database, FlaskConical, Settings, Waves } from "lucide-react";

import { useServerState } from "./api/events";
import { JobBar } from "./components/JobBar";
import { Banner } from "./components/ui";
import { useDarkMode } from "./lib/theme";
import { enabledTargets, useReadySamples, useStore, type View } from "./store";
import { DatasetsView } from "./views/DatasetsView";
import { ResultsView } from "./views/ResultsView";
import { SettingsView } from "./views/SettingsView";
import { TargetsView } from "./views/TargetsView";
import { WindowsView } from "./views/WindowsView";

const STEPS: { view: View; label: string; icon: typeof Database }[] = [
  { view: "datasets", label: "Datasets", icon: Database },
  { view: "targets", label: "Targets", icon: FlaskConical },
  { view: "windows", label: "RT windows", icon: Crosshair },
  { view: "results", label: "Results", icon: BarChart3 },
];

export function App() {
  useServerState();
  useDarkMode();
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const toast = useStore((s) => s.toast);
  const ready = useReadySamples().length;
  const targets = useStore((s) => s.targets);
  const windows = useStore((s) => s.windows);

  const done: Record<View, boolean> = {
    datasets: ready > 0,
    targets: enabledTargets(targets).length > 0,
    windows: windows.length >= 2,
    results: false,
    settings: false,
  };
  const hint: Partial<Record<View, string>> = {
    datasets: ready ? `${ready} ready` : undefined,
    targets: targets.length ? `${enabledTargets(targets).length}` : undefined,
    windows: windows.length ? `${windows.length}` : undefined,
  };

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="Workflow">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <Waves size={16} />
          </span>
          LCTIMSpy
        </div>
        {STEPS.map((s, i) => (
          <button
            key={s.view}
            className={`nav-item ${done[s.view] && view !== s.view ? "nav-done" : ""}`}
            aria-current={view === s.view ? "page" : undefined}
            onClick={() => setView(s.view)}
          >
            <span className="nav-step" aria-hidden>
              {i + 1}
            </span>
            {s.label}
            {hint[s.view] && <span className="nav-hint">{hint[s.view]}</span>}
          </button>
        ))}
        <div className="nav-spacer" />
        <button
          className="nav-item"
          aria-current={view === "settings" ? "page" : undefined}
          onClick={() => setView("settings")}
        >
          <Settings size={16} aria-hidden style={{ marginLeft: 2, marginRight: 2 }} />
          Settings
        </button>
      </nav>
      <main className="main" id="main">
        {view === "datasets" && <DatasetsView />}
        {view === "targets" && <TargetsView />}
        {view === "windows" && <WindowsView />}
        {view === "results" && <ResultsView />}
        {view === "settings" && <SettingsView />}
      </main>
      <JobBar />
      {toast && (
        <div className="toast" role="status" aria-live="polite">
          <Banner kind={toast.kind === "err" ? "err" : "info"} title={toast.kind === "err" ? "Something went wrong" : "Done"}>
            {toast.text}
          </Banner>
        </div>
      )}
    </div>
  );
}
