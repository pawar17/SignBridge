import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served from /SignBridge/ on GitHub Pages; "/" locally
export default defineConfig({
  base: process.env.BASE_PATH || "/",
  plugins: [react()],
});
