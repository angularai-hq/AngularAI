// Every item is a merged pull request or a commit on main. Dates are Europe/Bratislava (CEST).
export type Update = { date: string; title: string; href: string; ref: string };

const PR = (n: number) => `https://github.com/angularai-hq/AngularAI/pull/${n}`;
const C = (sha: string) => `https://github.com/angularai-hq/AngularAI/commit/${sha}`;

export const UPDATES: Update[] = [
  { date: "2026-10-20", title: "The site gets an MCP section", href: PR(14), ref: "#14" },
  { date: "2026-10-20", title: "The Angular AI MCP server", href: PR(13), ref: "#13" },
  { date: "2026-10-19", title: "Multiple cortexes talking", href: PR(12), ref: "#12" },
  { date: "2026-10-19", title: "Recall by meaning, with an optional embeddings API", href: PR(11), ref: "#11" },
  { date: "2026-10-19", title: "The budget counts CJK, Hangul and emoji as tokens", href: PR(10), ref: "#10" },
  { date: "2026-10-19", title: "The agent is told it sees files only through tools", href: PR(9), ref: "#9" },
  { date: "2026-10-18", title: "Adversarial agent tests, and the bugs they found", href: PR(8), ref: "#8" },
  { date: "2026-10-18", title: "Hypothesis property tests, and the bugs they found", href: PR(7), ref: "#7" },
  { date: "2026-10-18", title: "Provider timeouts, bounded retries and typed errors", href: PR(6), ref: "#6" },
  { date: "2026-10-18", title: "The cortex stays readable through crashes and bad lines", href: PR(5), ref: "#5" },
  { date: "2026-10-18", title: "Provider errors are named, and a secret is refused before the key", href: PR(4), ref: "#4" },
  { date: "2026-10-17", title: "CI runs the gates on every pull request", href: PR(3), ref: "#3" },
  { date: "2026-10-17", title: "More secrets are seen, and fewer words are taken for secrets", href: PR(2), ref: "#2" },
  { date: "2026-10-17", title: "The agent stays inside its directory", href: PR(1), ref: "#1" },
  { date: "2026-10-17", title: "An installer for the angularai command", href: C("07748fb"), ref: "07748fb" },
  { date: "2026-10-17", title: "The agent can work a directory", href: C("dd0c2d3"), ref: "dd0c2d3" },
  { date: "2026-10-17", title: "A public site", href: C("e2d1536"), ref: "e2d1536" },
  { date: "2026-10-17", title: "An agent's memory, kept outside the model window", href: C("acdee6c"), ref: "acdee6c" },
];

export const fmtDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" });

export type Release = { title: string; date: string; href: string; art: string; alt: string; summary: string };

export const RELEASES: Release[] = [
  {
    title: "The MCP server",
    date: "2026-10-20",
    href: "/mcp/",
    art: "mcp",
    alt: "Coloured ribbons converging on one white cube.",
    summary: "Any MCP client can remember, recall and forget in the same cortex.",
  },
  {
    title: "Cortexes that talk",
    date: "2026-10-19",
    href: "/platform/#talk",
    art: "cortexes",
    alt: "Two crystal towers joined by one yellow ribbon carrying a card.",
    summary: "Two cortexes take turns. Only a message crosses between them.",
  },
  {
    title: "Recall by meaning",
    date: "2026-10-19",
    href: "/platform/#recall",
    art: "recall",
    alt: "Swirling clusters of coloured dots on a dark field.",
    summary: "An optional embeddings API finds a trace that shares no word with the question.",
  },
  {
    title: "The agent in one folder",
    date: "2026-10-17",
    href: "/platform/#agent",
    art: "agent",
    alt: "Coloured blocks arranged inside an open glass box.",
    summary: "One tool a pass, inside one directory. A path that leaves it is refused.",
  },
];
