// The five layers, in the words of the README. Names come from Thameleon's CLASP model.
export const LAYERS = [
  {
    id: "cortex",
    name: "Cortex",
    line: "Long-term memory, on disk.",
    body: "Traces and episodes are lines in a file on the machine that asked. A superseded line stays in the file and leaves the workspace. A failed call writes the episode and no traces.",
  },
  {
    id: "membrane",
    name: "Membrane",
    line: "A fixed token budget for what reaches the model.",
    body: "Each call is packed to a fixed budget. Attention ranks the file against the words of the question, and the workspace takes only what fits. The file may outgrow the budget. The call may not.",
  },
  {
    id: "lock",
    name: "Value lock",
    line: "Secrets are blocked before they leave the machine.",
    body: "A secret in the input is refused before the call. A phrase you forbid, or a secret, in the reply or in the tool it asks for, holds it. You see one sentence. The held text is not stored.",
  },
  {
    id: "agent",
    name: "Agent",
    line: "A small agent that works inside one folder.",
    body: "One tool a pass, inside one directory: list, read, search, edit, write, run. A path that leaves the directory is refused. run is off unless you turn it on.",
  },
  {
    id: "mcp",
    name: "MCP connector",
    line: "Any MCP client shares the same memory.",
    body: "angularai-mcp is the cortex as MCP tools, for Claude Desktop, Claude Code, Codex, Cursor, or any MCP client. It runs no model.",
  },
] as const;
