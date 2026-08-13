import { defineConfig } from "vite";

// Transformers.js ships its own ESM plus onnxruntime-web wasm assets. Excluding
// it from Vite dependency pre-bundling avoids Vite trying to rewrite the wasm
// glue code, which is the most common cause of "failed to load model" errors.
export default defineConfig({
  // Relative base so the built site works from any path, including a GitHub
  // Pages project subpath like /copilot-unplugged/, without hardcoding the repo
  // name. Local dev (served at /) is unaffected.
  base: "./",
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  server: {
    port: 5173,
    open: false,
  },
});
