// Copy the static export (./out) into ../docs, which GitHub Pages serves from main:/docs.
// Files listed in KEEP are hand-kept in docs/ and are never touched.
import { cpSync, existsSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

const out = path.resolve("out");
const docs = path.resolve("..", "docs");
const KEEP = new Set([
  "install.sh", // served at /AngularAI/install.sh: the README install line depends on it
  "angularai.pdf", // the paper
  "build_paper.py", // builds the paper (writes docs/angularai.pdf)
  "logo.png", // README image
]);

if (!existsSync(path.join(out, "index.html"))) {
  console.error("No ./out/index.html. Run `pnpm build` first.");
  process.exit(1);
}
for (const name of readdirSync(docs)) {
  if (!KEEP.has(name)) rmSync(path.join(docs, name), { recursive: true, force: true });
}
for (const name of readdirSync(out)) {
  if (KEEP.has(name)) continue;
  cpSync(path.join(out, name), path.join(docs, name), { recursive: true });
}
writeFileSync(path.join(docs, ".nojekyll"), ""); // let Pages serve the _next/ folder
console.log(`docs/ updated from out/ (kept: ${[...KEEP].join(", ")})`);
