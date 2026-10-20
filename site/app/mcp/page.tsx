import type { Metadata } from "next";
import { PageShell, Md } from "@/components/PageShell";
import { RailSection } from "@/components/Section";
import { CodeBlock } from "@/components/CodeBlock";
import { readRepo, render, section } from "@/lib/md";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "MCP",
  description: "angularai-mcp: the Angular AI cortex as MCP tools for Claude Desktop, Claude Code, Codex, Cursor, or any MCP client. remember, recall, forget, status, cortexes.",
  alternates: { canonical: `${SITE.url}/mcp/` },
};

export default function Mcp() {
  const pkg = readRepo("packages/angularai-mcp/README.md");
  const intro = pkg.split("\n## ")[0].replace(/^# .*\n/, "").trim();
  const run = section(pkg, "Run").replace(/^```bash\nuvx[^\n]*\n```\n*/, "");
  return (
    <PageShell
      label="MCP"
      title="One memory for every MCP client."
      lede={
        <p>
          The cortex as an MCP server. Five tools: <code>remember</code>, <code>recall</code>, <code>forget</code>, <code>status</code>, <code>cortexes</code>. Stdio by default. <code>--http</code> serves streamable HTTP on 127.0.0.1.
        </p>
      }
    >
      <RailSection id="run" label="Run" className="!pt-4">
        <div className="max-w-[760px]">
          <CodeBlock code={SITE.mcpRun} />
          <p className="mt-4 text-[15px]">
            Claude Desktop, Claude Code, Codex, Cursor, or any MCP client. <a className="inline" href="#setup">Setup for each client</a>.
          </p>
        </div>
        <Md html={render(intro)} className="mt-8" />
        <Md html={render(run)} className="mt-4" />
      </RailSection>
      <RailSection id="tools" label="Tools" className="!pt-0"><Md html={render(section(pkg, "Tools"))} /></RailSection>
      <RailSection id="setup" label="Setup" className="!pt-0"><Md html={render(section(pkg, "Setup"))} /></RailSection>
      <RailSection id="http" label="HTTP" className="!pt-0"><Md html={render(section(pkg, "HTTP"))} /></RailSection>
      <RailSection id="errors" label="Errors" className="!pt-0"><Md html={render(section(pkg, "Errors"))} /></RailSection>
      <RailSection id="verify" label="Verify" className="!pt-0"><Md html={render(section(pkg, "Verify"))} /></RailSection>
    </PageShell>
  );
}
