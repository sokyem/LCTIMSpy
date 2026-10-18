import { useEffect, useRef, useState } from "react";

export interface FetchState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

/**
 * Run ``fn`` whenever ``key`` changes, after ``delay`` ms of quiet. A newer key
 * aborts the in-flight request, so a slow answer for an old window can never
 * overwrite the answer for the current one. ``key = null`` clears the state.
 */
export function useFetch<T>(key: string | null, fn: (signal: AbortSignal) => Promise<T>, delay = 0): FetchState<T> {
  const [state, setState] = useState<FetchState<T>>({ data: null, error: null, loading: false });
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (key == null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true }));
    const timer = setTimeout(() => {
      fnRef.current(ctrl.signal).then(
        (data) => !ctrl.signal.aborted && setState({ data, error: null, loading: false }),
        (error: Error) =>
          !ctrl.signal.aborted && error.name !== "AbortError" && setState((s) => ({ ...s, error, loading: false })),
      );
    }, delay);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [key, delay]);

  return state;
}
