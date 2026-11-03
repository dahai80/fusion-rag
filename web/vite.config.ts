import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Standalone dev proxies the fusion-rag API surface to :11436 so dev mirrors
// the embedded gateway /shell/fusion-rag/ layout (same-origin).
// FUSION_DEV_BACKEND overrides. Dev server uses the vite default port
// (dev-only tooling, not a registered service port).
const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
const backend = proc?.env?.FUSION_DEV_BACKEND ?? "http://127.0.0.1:11436";

const apiPaths = ["/kb", "/health", "/metrics", "/ready"];

export default defineConfig({
    plugins: [react()],
    // Embedded deployment serves dist through the gateway sub-path
    // /shell/fusion-rag/ — relative base works in both embedded and
    // standalone modes.
    base: "./",
    server: {
        proxy: Object.fromEntries(
            apiPaths.map((p) => [p, { target: backend, changeOrigin: true }]),
        ),
    },
    preview: {
        proxy: Object.fromEntries(
            apiPaths.map((p) => [p, { target: backend, changeOrigin: true }]),
        ),
    },
});
