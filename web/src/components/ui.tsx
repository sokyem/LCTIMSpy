import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";

export function Banner({
  kind = "info",
  title,
  children,
  action,
}: {
  kind?: "info" | "warn" | "err";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const Icon = kind === "err" ? XCircle : kind === "warn" ? AlertTriangle : Info;
  return (
    <div className={`banner banner-${kind}`} role={kind === "info" ? "status" : "alert"}>
      <Icon size={18} aria-hidden />
      <div className="stack" style={{ gap: 2, flex: 1 }}>
        <strong>{title}</strong>
        {children && <div className="muted">{children}</div>}
      </div>
      {action}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      {icon}
      <strong style={{ color: "var(--text)" }}>{title}</strong>
      {children}
    </div>
  );
}

/** A number input that edits a local draft and commits on blur/Enter, so typing
 * "18." never fires a query for 18 or rejects a half-typed value. */
export function NumberInput({
  value,
  onCommit,
  step = 0.01,
  min,
  max,
  label,
  className = "input input-sm num",
  width,
}: {
  value: number | null;
  onCommit: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  label: string;
  className?: string;
  width?: number;
}) {
  const [draft, setDraft] = useState(value == null ? "" : String(value));
  const focused = useRef(false);
  useEffect(() => {
    if (!focused.current) setDraft(value == null ? "" : String(value));
  }, [value]);
  const n = Number(draft);
  const invalid = draft.trim() === "" || !Number.isFinite(n) || (min != null && n < min) || (max != null && n > max);
  const commit = () => {
    if (!invalid && n !== value) onCommit(n);
    else if (invalid) setDraft(value == null ? "" : String(value));
  };
  return (
    <input
      className={className}
      style={width ? { width } : undefined}
      type="number"
      inputMode="decimal"
      step={step}
      min={min}
      max={max}
      value={draft}
      aria-label={label}
      aria-invalid={invalid}
      onFocus={() => (focused.current = true)}
      onBlur={() => {
        focused.current = false;
        commit();
      }}
      onChange={(e) => setDraft(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "Escape") {
          setDraft(value == null ? "" : String(value));
          (e.target as HTMLInputElement).blur();
        }
      }}
    />
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function StatusIcon({ ok }: { ok: boolean }) {
  return ok ? (
    <CheckCircle2 size={14} aria-hidden style={{ color: "var(--success)" }} />
  ) : (
    <XCircle size={14} aria-hidden style={{ color: "var(--danger)" }} />
  );
}
