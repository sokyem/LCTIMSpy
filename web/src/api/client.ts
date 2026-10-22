// Thin typed wrapper over fetch. Auth rides on the HttpOnly session cookie set
// when the app was opened with ?token=..., so no token handling lives here.
import type { components } from "./schema";

type S = components["schemas"];
export type Dataset = S["DatasetOut"];
export type Job = S["JobOut"];
export type Capabilities = S["CapabilitiesOut"];
export type FsEntry = S["FsEntry"];
export type FsListing = S["FsListing"];
export type ApiTarget = S["Target"];
export type Query = S["Query"];
export type Stats = S["Stats"];
export type Series = S["SeriesOut"];
export type MobilogramSeries = S["MobilogramOut"];
export type Peak = S["PeakOut"];

// Row shapes of the analysis tables (the server types these as free-form dicts).
export interface DiffRow {
  sample_id: string;
  sample_name: string;
  target_mz: number;
  label: string | null;
  window_a: number;
  window_b: number;
  avg_mobility_a: number | null;
  avg_mobility_b: number | null;
  mobility_difference: number | null;
  peak_mobility_a: number;
  peak_mobility_b: number;
  peak_mobility_difference: number;
}
export interface ExcludedRow {
  sample_id: string;
  sample_name: string;
  target_mz: number;
  label: string | null;
  reason: string;
}
export interface PairRow {
  group_a: number;
  group_b: number;
  n_a: number;
  n_b: number;
  statistic: number | null;
  raw_p_value: number | null;
  adjusted_p_value: number | null;
  reject: boolean;
}
export interface MobilityDiff {
  per_window: Record<string, unknown>[];
  differences: DiffRow[];
  excluded: ExcludedRow[];
}
export interface Compare {
  differences: MobilityDiff;
  pairs: PairRow[];
  data: DiffRow[];
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public remediation: string | null = null,
    public body: unknown = null,
  ) {
    super(message);
  }
}

function describe(body: unknown, fallback: string): { message: string; remediation: string | null } {
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    const remediation = typeof b.remediation === "string" ? b.remediation : null;
    const detail = b.detail;
    if (typeof detail === "string") return { message: detail, remediation };
    if (Array.isArray(detail)) {
      // pydantic validation errors
      const msgs = detail.map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)));
      return { message: msgs.join("; "), remediation };
    }
    if (detail && typeof detail === "object" && "message" in detail) {
      return { message: String((detail as { message: unknown }).message), remediation };
    }
  }
  return { message: fallback, remediation: null };
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
    signal,
  });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  const parsed = text ? safeJson(text) : null;
  if (!res.ok) {
    const { message, remediation } = describe(parsed, `${res.status} ${res.statusText}`);
    throw new ApiError(res.status, message, remediation, parsed);
  }
  return parsed as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => request<T>("GET", path, undefined, signal),
  post: <T>(path: string, body: unknown, signal?: AbortSignal) => request<T>("POST", path, body, signal),
  patch: <T>(path: string, body: unknown) => request<T>("PATCH", path, body),
  del: <T>(path: string) => request<T>("DELETE", path),
};

/** POST to /api/export and hand the response to the browser as a file download. */
export async function download(kind: string, format: "csv" | "html" | "json", req: unknown): Promise<void> {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, format, request: req }),
    credentials: "same-origin",
  });
  if (!res.ok) {
    const { message, remediation } = describe(safeJson(await res.text()), res.statusText);
    throw new ApiError(res.status, message, remediation);
  }
  const blob = await res.blob();
  const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") ?? "")?.[1] ?? `export.${format}`;
  saveBlob(blob, name);
}

export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---- discovery (server returns free-form JSON for these) --------------------------------

export interface DiscoverySpecies {
  rt_peak: number;
  population: number;
  n_populations: number;
  kind: string;
  rt_apex: number;
  rt_start: number;
  rt_end: number;
  rt_area: number | null;
  signal: number | null;
  height: number;
  mobility: number | null;
  mobility_fwhm: number | null;
  snr: number | null;
  psm_count: number;
  mod_keys: string[];
  feature_ids: number[];
  flags: string[];
}
export interface Criterion {
  value: number | null;
  threshold: number;
  met: boolean;
}
export interface DiscoveryCandidate {
  candidate_id: number;
  route: string;
  group: string | number;
  charge: number;
  mono_mz: number;
  neutral_mass: number;
  mod_key: string | null;
  mod_keys?: string[];
  classification: string;
  category: string;
  sequence_status: string;
  delta_rt: number | null;
  rt_resolution: number | null;
  rt_separation: string;
  delta_mobility: number | null;
  mobility_rel_diff_pct: number | null;
  mobility_reference: string;
  relative_signal_fraction_a: number | null;
  species_a: DiscoverySpecies;
  species_b: DiscoverySpecies | null;
  criteria: Record<string, Criterion>;
  flags: string[];
  status: "candidate" | "excluded";
  exclusion_reasons: string[];
  msms_status?: string;
  msms_cosine?: number | null;
}
export interface DiscoveryCounts {
  total: number;
  candidates: number;
  excluded: number;
  by_class: Record<string, number>;
  by_category: Record<string, number>;
}
export interface DiscoveryRunSummary {
  id: string;
  created: number;
  dataset: { name: string; source_path: string };
  counts: DiscoveryCounts;
  psms: { engine: string; used: number; total: number; note: string } | null;
  n_features: number;
  notes: string[];
}
export interface DiscoveryRun {
  id: string;
  meta: { counts: DiscoveryCounts; notes: string[]; config: Record<string, unknown>; mobility: { coordinate: string | null } } & Record<string, unknown>;
  candidates: DiscoveryCandidate[];
}
export interface MsmsSide {
  n_assigned: number;
  n_unambiguous: number;
  best_precursor: number | null;
  best_ambiguous: boolean | null;
  within_similarity_median: number | null;
  spectra: { precursor_id: number; rt: number; mobility: number | null; ambiguous: boolean; reasons: string[]; n_peaks: number }[];
}
export interface CandidateDetail {
  candidate: DiscoveryCandidate;
  msms: {
    a: MsmsSide | null;
    b: MsmsSide | null;
    similarity: { cosine: number | null; spectral_angle: number | null; n_matched: number; pearson: number | null } | null;
    annotation: Record<string, { a: { fraction: number | null; n_ions: number } | null; b: { fraction: number | null; n_ions: number } | null }>;
    differential_fragments: { mz: number; a: number; b: number; log_ratio: number | null }[];
  };
  detail: {
    coordinate: string | null;
    chromatogram: { rt: number[]; intensity: number[]; isotopes_summed: number; ppm: number };
    species: (DiscoverySpecies & { name: string })[];
    mobilograms: { species: string; x: number[]; y: number[]; centre: number | null; fwhm: number | null }[];
    map: { rt: number[]; mobility: number[]; z: number[][] } | null;
    ms1: { species: string; mz: number[]; intensity: number[]; averagine_mz: number[]; averagine_relative: number[] }[];
    mirror: { a_mz: number[]; a_intensity: number[]; b_mz: number[]; b_intensity: number[]; matches: number[][] } | null;
    labels: { measured: string[]; calculated: string[] };
  } | null;
  note: string | null;
}

/** GET a file the server offers as a download and save it. */
export async function downloadGet(path: string): Promise<void> {
  const res = await fetch(`/api${path}`, { credentials: "same-origin" });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  const name = /filename="([^"]+)"/.exec(res.headers.get("content-disposition") ?? "")?.[1] ?? "download";
  saveBlob(await res.blob(), name);
}
