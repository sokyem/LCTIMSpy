/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: run `lctimspy serve --port 8765 --no-token`, then `npm run dev`.
const api = process.env.LCTIMS_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  build: {
    // Served by FastAPI and committed, so `pip install` needs no Node.
    outDir: "../LCTIMSpy/server/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
  },
  server: {
    proxy: {
      "/api": {
        target: api,
        changeOrigin: true,
        // The server refuses foreign Origins; the dev page's origin is the proxy's, not the API's.
        configure: (proxy) => proxy.on("proxyReq", (req) => req.removeHeader("origin")),
      },
    },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
