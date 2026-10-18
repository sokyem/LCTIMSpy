export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return u === 0 ? `${v} B` : `${v.toFixed(1)} ${units[u]}`;
}

export function fmtEta(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${String(r).padStart(2, "0")}s` : `${r}s`;
}

export const fmtMz = (mz: number) => mz.toFixed(4);
export const fmtRt = (rt: number) => rt.toFixed(2);

export function fmtMobility(v: number | null | undefined, digits = 4): string {
  return v == null || !Number.isFinite(v) ? "–" : v.toFixed(digits);
}

export function fmtP(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "–";
  return p < 1e-4 ? p.toExponential(1) : p.toPrecision(2);
}

export function stars(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "";
  if (p <= 1e-4) return "****";
  if (p <= 1e-3) return "***";
  if (p <= 1e-2) return "**";
  if (p <= 0.05) return "*";
  return "ns";
}
