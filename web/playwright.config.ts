import { defineConfig } from "@playwright/test";
import path from "node:path";

// Runs the built UI (LCTIMSpy/server/static) against the demo server with
// synthetic extracted data. Uses the installed Edge, so no browser download.
const PORT = 8799;
const venv = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
// Absolute and quoted: on Windows the command runs through cmd.exe.
const python = process.env.LCTIMS_PYTHON ?? path.resolve(process.cwd(), "..", venv);

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    channel: process.env.PW_CHANNEL ?? "msedge",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    // The export test checks the response; tracking the browser download
    // itself errors in worker teardown on Edge.
    acceptDownloads: false,
  },
  webServer: {
    command: `"${python}" -m tests.e2e.demo_server --port ${PORT} --token e2e`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
