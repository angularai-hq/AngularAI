export const SITE = {
  name: "Angular AI",
  url: "https://angularai-hq.github.io/AngularAI",
  repo: "https://github.com/angularai-hq/AngularAI",
  basePath: "/AngularAI",
  install: "curl -fsSL https://angularai-hq.github.io/AngularAI/install.sh | sh",
  mcpRun:
    'uvx --from "git+https://github.com/angularai-hq/AngularAI#subdirectory=packages/angularai-mcp" angularai-mcp',
  description:
    "Angular AI is a lightweight Thameleon: a cortex, a membrane, a value lock, a small agent and an MCP connector on top of the model you already use. Free and open source.",
} as const;

/** Prefix a public asset path with the basePath (plain <img>/<video> do not get it automatically). */
export const asset = (p: string) => `${SITE.basePath}${p.startsWith("/") ? p : `/${p}`}`;

export const abs = (p: string) => `${SITE.url}${p}`;
