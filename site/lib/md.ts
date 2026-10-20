import "server-only";
import { readFileSync } from "node:fs";
import path from "node:path";
import { marked } from "marked";

// Content is read from the repository's own READMEs at build time, so the site and the docs cannot drift.
const ROOT = path.join(process.cwd(), "..");

export function readRepo(rel: string) {
  return readFileSync(path.join(ROOT, rel), "utf8");
}

/** The body of one `## Heading` section (without the heading). */
export function section(md: string, heading: string, level = 2) {
  const hashes = "#".repeat(level);
  const lines = md.split("\n");
  const start = lines.findIndex((l) => l.trim() === `${hashes} ${heading}`);
  if (start < 0) throw new Error(`Section not found: ${heading}`);
  const out: string[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const m = /^(#+)\s/.exec(lines[i]);
    if (m && m[1].length <= level) break;
    out.push(lines[i]);
  }
  return out.join("\n").trim();
}

const RELINK: [RegExp, string][] = [
  [/href="packages\/angularai-mcp"/g, 'href="https://github.com/angularai-hq/AngularAI/tree/main/packages/angularai-mcp"'],
  [/href="docs\/angularai\.pdf"/g, 'href="/AngularAI/angularai.pdf"'],
];

/** Visible copy spells the name "Angular AI"; code, URLs and attributes keep "AngularAI". */
function brandText(html: string) {
  const parts = html.split(/(<[^>]+>)/);
  let inCode = 0;
  return parts
    .map((p) => {
      if (p.startsWith("<")) {
        if (/^<(code|pre)[\s>]/.test(p)) inCode++;
        else if (/^<\/(code|pre)>/.test(p)) inCode = Math.max(0, inCode - 1);
        return p;
      }
      return inCode ? p : p.replace(/(?<![/\w-])AngularAI\b/g, "Angular AI");
    })
    .join("");
}

export function render(md: string) {
  let html = marked.parse(md, { async: false, gfm: true }) as string;
  for (const [re, to] of RELINK) html = html.replace(re, to);
  // Scrollable code is keyboard reachable.
  html = html.replace(/<pre>/g, '<pre tabindex="0">');
  return brandText(html);
}
