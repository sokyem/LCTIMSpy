import { useEffect } from "react";

import { useStore } from "../store";
import { api, type Dataset, type Job } from "./client";

/** Load server state, then follow the multiplexed SSE stream.
 *
 * EventSource reconnects by itself and resends Last-Event-ID, so a dropped
 * connection replays what it missed. A ``resync`` event (the gap was older than
 * the server's buffer) or a reconnect after an error triggers a full refetch.
 */
export function useServerState(): void {
  useEffect(() => {
    const s = useStore.getState();
    let cancelled = false;

    const refetch = async () => {
      try {
        const [datasets, jobs, caps] = await Promise.all([
          api.get<Dataset[]>("/datasets"),
          api.get<Job[]>("/jobs"),
          api.get<typeof s.caps>("/capabilities"),
        ]);
        if (cancelled) return;
        const st = useStore.getState();
        st.setDatasets(datasets);
        st.setJobs(jobs);
        if (caps) st.setCaps(caps);
        // Forget selections of datasets the server no longer knows.
        for (const id of st.selected) if (!datasets.some((d) => d.id === id)) st.toggleSelected(id, false);
      } catch (e) {
        if (!cancelled) useStore.getState().notify("err", `Cannot reach the LCTIMSpy server: ${String(e)}`);
      }
    };

    void refetch();
    const source = new EventSource("/api/events");
    let hadError = false;
    const on = (type: string, fn: (data: unknown) => void) =>
      source.addEventListener(type, (e) => fn(JSON.parse((e as MessageEvent).data)));

    on("job", (d) => useStore.getState().upsertJob(d as Job));
    on("progress", (d) => {
      const { job_id, ...progress } = d as { job_id: string } & Record<string, unknown>;
      useStore.getState().setJobProgress(job_id, progress);
    });
    on("dataset", (d) => useStore.getState().upsertDataset(d as Dataset));
    on("dataset_removed", (d) => useStore.getState().removeDataset((d as { id: string }).id));
    on("resync", () => void refetch());
    source.onerror = () => {
      hadError = true;
    };
    source.onopen = () => {
      if (hadError) {
        hadError = false;
        void refetch();
      }
    };
    return () => {
      cancelled = true;
      source.close();
    };
  }, []);
}
