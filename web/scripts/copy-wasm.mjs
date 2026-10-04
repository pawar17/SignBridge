// Copy MediaPipe's WASM runtime from the npm package into public/, so the app
// serves it itself instead of depending on a CDN.
import { cpSync, mkdirSync, readdirSync } from "node:fs";
const src = "node_modules/@mediapipe/tasks-vision/wasm";
mkdirSync("public/wasm", { recursive: true });
for (const f of readdirSync(src)) if (!f.includes("module")) cpSync(`${src}/${f}`, `public/wasm/${f}`);
