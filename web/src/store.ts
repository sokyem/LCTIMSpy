import { create } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";

import type { Capabilities, Dataset, Job, Query, Stats } from "./api/client";
import { freeSlot, MAX_WINDOWS } from "./lib/palette";
import type { RtWindow } from "./lib/relayout";

export type View = "datasets" | "targets" | "windows" | "results" | "settings" | "discovery";
export type Theme = "system" | "light" | "dark";

export interface Target {
  mz: number;
  label: string | null;
  enabled: boolean;
  /** Colour slot, held while enabled so toggling others never repaints this one. */
  slot: number | null;
}

export interface Session {
  version: 1;
  selected: string[];
  activeSample: string | null;
  targets: Target[];
  windows: RtWindow[];
  offsets: Record<string, number>;
  query: Required<Query>;
  stats: Required<Stats>;
  peptide: { sequence: string; charges: number[] };
}

export const DEFAULT_QUERY: Required<Query> = {
  tolerance: { value: 0.02, unit: "Da" },
  mobilogram_sigma: null,
  baseline_correction: true,
  intensity_threshold: 100,
  threshold_on: "corrected",
};

export const DEFAULT_STATS: Required<Stats> = {
  metric: "mobility_difference",
  correction: "bonferroni",
  alpha: 0.05,
  min_samples: 3,
  equal_var: true,
};

const emptySession = (): Session => ({
  version: 1,
  selected: [],
  activeSample: null,
  targets: [],
  windows: [],
  offsets: {},
  query: DEFAULT_QUERY,
  stats: DEFAULT_STATS,
  peptide: { sequence: "", charges: [1, 2] },
});

interface State extends Session {
  view: View;
  theme: Theme;
  armed: number | null;
  // live server state (not persisted)
  datasets: Record<string, Dataset>;
  jobs: Record<string, Job>;
  caps: Capabilities | null;
  toast: { kind: "ok" | "err"; text: string } | null;
  discoveryVersion: number;
  discoveryRun: string | null;

  setView: (v: View) => void;
  setTheme: (t: Theme) => void;
  setDatasets: (list: Dataset[]) => void;
  upsertDataset: (d: Dataset) => void;
  removeDataset: (id: string) => void;
  setJobs: (list: Job[]) => void;
  upsertJob: (j: Job) => void;
  setJobProgress: (id: string, progress: Job["progress"]) => void;
  setCaps: (c: Capabilities) => void;
  notify: (kind: "ok" | "err", text: string) => void;

  toggleSelected: (id: string, on?: boolean) => void;
  setActiveSample: (id: string | null) => void;
  addTargets: (ts: { mz: number; label: string | null }[], enable?: boolean) => string | null;
  setTargetEnabled: (index: number, on: boolean) => string | null;
  updateTarget: (index: number, patch: Partial<Pick<Target, "label" | "mz">>) => void;
  removeTarget: (index: number) => void;
  clearTargets: () => void;
  setWindow: (index: number, w: RtWindow) => void;
  setWindows: (ws: RtWindow[]) => void;
  removeWindow: (index: number) => void;
  arm: (index: number | null) => void;
  fillArmed: (w: RtWindow) => void;
  setOffset: (id: string, minutes: number) => void;
  setQuery: (q: Partial<Required<Query>>) => void;
  setStats: (s: Partial<Required<Stats>>) => void;
  setPeptide: (p: Partial<Session["peptide"]>) => void;
  loadSession: (s: Session) => void;
  bumpDiscovery: (runId?: string | null) => void;
  setDiscoveryRun: (runId: string | null) => void;
  /** Hand a discovery candidate to the CID-TIMS localization workflow. */
  handoffToLocalization: (targets: { mz: number; label: string | null }[], windows: RtWindow[],
                          peptide: string, charges: number[]) => string | null;
}

const LIMIT_MSG = "At most 8 targets can be plotted at once (one colour each). Disable one first.";

// localStorage can be missing or throw (private windows, blocked site data).
const safeStorage: StateStorage = {
  getItem: (k) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  setItem: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* not persisted; the app still works */
    }
  },
  removeItem: (k) => {
    try {
      localStorage.removeItem(k);
    } catch {
      /* ignore */
    }
  },
};

export const useStore = create<State>()(
  persist(
    (set, get) => ({
      ...emptySession(),
      view: "datasets",
      theme: "system",
      armed: 0,
      datasets: {},
      jobs: {},
      caps: null,
      toast: null,
      discoveryVersion: 0,
      discoveryRun: null,

      setView: (view) => set({ view }),
      setTheme: (theme) => set({ theme }),
      setDatasets: (list) => set({ datasets: Object.fromEntries(list.map((d) => [d.id, d])) }),
      upsertDataset: (d) => set((s) => ({ datasets: { ...s.datasets, [d.id]: d } })),
      removeDataset: (id) =>
        set((s) => {
          const { [id]: _gone, ...rest } = s.datasets;
          return {
            datasets: rest,
            selected: s.selected.filter((x) => x !== id),
            activeSample: s.activeSample === id ? null : s.activeSample,
          };
        }),
      setJobs: (list) => set({ jobs: Object.fromEntries(list.map((j) => [j.id, j])) }),
      upsertJob: (j) => set((s) => ({ jobs: { ...s.jobs, [j.id]: { ...s.jobs[j.id], ...j } } })),
      setJobProgress: (id, progress) =>
        set((s) => (s.jobs[id] ? { jobs: { ...s.jobs, [id]: { ...s.jobs[id], progress } } } : {})),
      setCaps: (caps) => set({ caps }),
      notify: (kind, text) => {
        set({ toast: { kind, text } });
        setTimeout(() => {
          if (get().toast?.text === text) set({ toast: null });
        }, kind === "err" ? 8000 : 3500);
      },

      toggleSelected: (id, on) =>
        set((s) => {
          const has = s.selected.includes(id);
          const want = on ?? !has;
          if (want === has) return {};
          const selected = want ? [...s.selected, id] : s.selected.filter((x) => x !== id);
          const activeSample =
            s.activeSample && selected.includes(s.activeSample) ? s.activeSample : (selected[0] ?? null);
          return { selected, activeSample };
        }),
      setActiveSample: (activeSample) => set({ activeSample }),

      addTargets: (incoming, enable = true) => {
        let refused = 0;
        set((s) => {
          const targets = [...s.targets];
          for (const t of incoming) {
            if (targets.some((x) => Math.abs(x.mz - t.mz) < 1e-6)) continue;
            const slot = enable ? freeSlot(targets.filter((x) => x.enabled).map((x) => x.slot)) : null;
            if (enable && slot == null) refused++;
            targets.push({ mz: t.mz, label: t.label, enabled: slot != null, slot });
          }
          return { targets };
        });
        return refused ? `${refused} target(s) added but not plotted: ${LIMIT_MSG}` : null;
      },
      setTargetEnabled: (index, on) => {
        const s = get();
        const t = s.targets[index];
        if (!t || t.enabled === on) return null;
        let slot: number | null = null;
        if (on) {
          slot = freeSlot(s.targets.filter((x) => x.enabled).map((x) => x.slot));
          if (slot == null) return LIMIT_MSG;
        }
        const targets = s.targets.map((x, i) => (i === index ? { ...x, enabled: on, slot } : x));
        set({ targets });
        return null;
      },
      updateTarget: (index, patch) =>
        set((s) => ({ targets: s.targets.map((x, i) => (i === index ? { ...x, ...patch } : x)) })),
      removeTarget: (index) => set((s) => ({ targets: s.targets.filter((_, i) => i !== index) })),
      clearTargets: () => set({ targets: [] }),

      setWindow: (index, w) =>
        set((s) => {
          const windows = [...s.windows];
          windows[index] = w;
          return { windows };
        }),
      setWindows: (windows) => set({ windows: windows.slice(0, MAX_WINDOWS), armed: null }),
      removeWindow: (index) =>
        set((s) => ({ windows: s.windows.filter((_, i) => i !== index), armed: null })),
      arm: (armed) => set({ armed }),
      fillArmed: (w) =>
        set((s) => {
          const windows = [...s.windows];
          const i = s.armed ?? Math.min(windows.length, MAX_WINDOWS - 1);
          windows[i] = w;
          // A, then B: the isomer workflow is two windows, so arm the second automatically.
          const next = windows.length < 2 ? windows.length : null;
          return { windows, armed: next };
        }),
      setOffset: (id, minutes) => set((s) => ({ offsets: { ...s.offsets, [id]: minutes } })),
      setQuery: (q) => set((s) => ({ query: { ...s.query, ...q } })),
      setStats: (st) => set((s) => ({ stats: { ...s.stats, ...st } })),
      setPeptide: (p) => set((s) => ({ peptide: { ...s.peptide, ...p } })),
      bumpDiscovery: (runId) =>
        set((s) => ({ discoveryVersion: s.discoveryVersion + 1, discoveryRun: runId ?? s.discoveryRun })),
      setDiscoveryRun: (discoveryRun) => set({ discoveryRun }),
      handoffToLocalization: (targets, windows, peptide, charges) => {
        set({ targets: [], windows: windows.slice(0, MAX_WINDOWS), armed: null,
              peptide: { sequence: peptide, charges } });
        const warning = get().addTargets(targets);
        set({ view: "targets" });
        return warning;
      },
      loadSession: (session) =>
        set({
          ...emptySession(),
          ...session,
          query: { ...DEFAULT_QUERY, ...session.query },
          stats: { ...DEFAULT_STATS, ...session.stats },
          armed: null,
        }),
    }),
    {
      name: "lctimspy-session",
      storage: createJSONStorage(() => safeStorage),
      partialize: (s) => ({
        version: s.version,
        selected: s.selected,
        activeSample: s.activeSample,
        targets: s.targets,
        windows: s.windows,
        offsets: s.offsets,
        query: s.query,
        stats: s.stats,
        peptide: s.peptide,
        view: s.view,
        theme: s.theme,
        discoveryRun: s.discoveryRun,
      }),
    },
  ),
);

/** The session part of the state, for saving to a .lctims.json file. */
export function currentSession(): Session {
  const s = useStore.getState();
  return {
    version: 1,
    selected: s.selected,
    activeSample: s.activeSample,
    targets: s.targets,
    windows: s.windows,
    offsets: s.offsets,
    query: s.query,
    stats: s.stats,
    peptide: s.peptide,
  };
}

/** Datasets selected for analysis that have a cache entry. */
export function useReadySamples(): Dataset[] {
  const selected = useStore((s) => s.selected);
  const datasets = useStore((s) => s.datasets);
  return selected.map((id) => datasets[id]).filter((d): d is Dataset => !!d && d.cache.status === "cached");
}

export function enabledTargets(targets: Target[]): Target[] {
  return targets.filter((t) => t.enabled);
}
