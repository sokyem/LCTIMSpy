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
