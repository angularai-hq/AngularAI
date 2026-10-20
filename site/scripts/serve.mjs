// Serve a static build under the /AngularAI basePath, like GitHub Pages does.
// Usage: node scripts/serve.mjs [dir=out] [port=4321]
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import path from "node:path";
import { createGzip } from "node:zlib";

const root = path.resolve(process.argv[2] ?? "out");
const port = Number(process.argv[3] ?? 4321);
const BASE = "/AngularAI";
const TYPES = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".json": "application/json", ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp", ".ico": "image/x-icon", ".mp4": "video/mp4", ".webm": "video/webm", ".woff2": "font/woff2", ".txt": "text/plain", ".xml": "application/xml", ".pdf": "application/pdf", ".sh": "text/plain; charset=utf-8" };

const file = (p) => { try { const s = statSync(p); return s.isFile() ? p : s.isDirectory() ? file(path.join(p, "index.html")) : null; } catch { return null; } };

createServer((req, res) => {
  const url = decodeURIComponent((req.url ?? "/").split("?")[0]);
  if (url === "/" ) { res.writeHead(302, { Location: `${BASE}/` }); return res.end(); }
  let hit = null;
  if (url.startsWith(`${BASE}/`) || url === BASE) {
    const rel = url.slice(BASE.length) || "/";
    const p = path.join(root, rel);
    if (p.startsWith(root)) hit = file(p);
    if (!hit && url === BASE) { res.writeHead(301, { Location: `${BASE}/` }); return res.end(); }
  }
  const status = hit ? 200 : 404;
  hit ??= path.join(root, "404.html");
  const size = statSync(hit).size;
  const type = TYPES[path.extname(hit)] ?? "application/octet-stream";
  const range = req.headers.range && /bytes=(\d*)-(\d*)/.exec(req.headers.range);
  if (range && status === 200) {
    const start = range[1] ? Number(range[1]) : 0;
    const end = range[2] ? Number(range[2]) : size - 1;
    res.writeHead(206, { "Content-Type": type, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1, "Accept-Ranges": "bytes" });
    return createReadStream(hit, { start, end }).pipe(res);
  }
  // GitHub Pages gzips text assets; do the same so local Lighthouse numbers are comparable.
  if (/text|javascript|json|xml|svg|manifest/.test(type) && /gzip/.test(req.headers["accept-encoding"] ?? "")) {
    res.writeHead(status, { "Content-Type": type, "Content-Encoding": "gzip", "Vary": "Accept-Encoding", "Cache-Control": "public, max-age=600" });
    return createReadStream(hit).pipe(createGzip()).pipe(res);
  }
  res.writeHead(status, { "Content-Type": type, "Content-Length": size, "Accept-Ranges": "bytes", "Cache-Control": "public, max-age=600" });
  createReadStream(hit).pipe(res);
}).listen(port, () => console.log(`http://localhost:${port}${BASE}/  (serving ${root})`));
