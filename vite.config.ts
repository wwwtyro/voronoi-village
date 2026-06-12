import { defineConfig } from "vite";

// Relative base so the build works from any mount point, including the
// GitHub Pages project subpath.
export default defineConfig({
  base: "./",
});
