import type { NextConfig } from "next";

// Static export for GitHub Pages: https://angularai-hq.github.io/AngularAI/
// `pnpm export:docs` builds into ./out and copies it into ../docs (see scripts/export-docs.mjs).
const config: NextConfig = {
  output: "export",
  basePath: "/AngularAI",
  trailingSlash: true,
  images: { unoptimized: true },
  poweredByHeader: false,
  reactStrictMode: true,
};

export default config;
