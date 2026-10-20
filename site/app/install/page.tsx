import type { Metadata } from "next";
import { PageShell, Md } from "@/components/PageShell";
import { RailSection } from "@/components/Section";
import { CodeBlock } from "@/components/CodeBlock";
import { LabelLink } from "@/components/Chevron";
import { readRepo, render, section } from "@/lib/md";
import { SITE, asset } from "@/lib/site";

export const metadata: Metadata = {
  title: "Install",
  description: "Install the angularai command with one line, wire the API you already have, and speak to the cortex.",
  alternates: { canonical: `${SITE.url}/install/` },
};

export default function Install() {
  const readme = readRepo("README.md");
  // The README's install section, minus its first code block (shown above with a copy button).
  const install = section(readme, "Install").replace(/^```bash\ncurl[^\n]*\n```\n*/, "");
  return (
    <PageShell
      label="Install"
      title="One line, then the API you already have."
      lede={<p>Then wire the API you already have. The key stays in the environment.</p>}
    >
      <RailSection id="command" label="Command" className="!pt-4">
        <div className="max-w-[760px]">
          <CodeBlock code={SITE.install} />
          <div className="md mt-6">
            <p>The script puts <code>angularai</code> on your PATH. It does not read or store a key. If <code>uv</code> is missing, it installs it first. Run it as your user, not as root. <code>ANGULARAI_INSTALL_PLAN=1</code> prints what it would do and changes nothing.</p>
            <p><a href={asset("/install.sh")}>Read install.sh</a></p>
          </div>
        </div>
      </RailSection>
      <RailSection id="wire" label="Wire" className="!pt-0">
        <Md html={render(install)} />
      </RailSection>
      <RailSection id="speak" label="Speak" className="!pt-0">
        <Md html={render(section(readme, "Speak"))} />
      </RailSection>
      <RailSection id="next" label="Next" className="!pt-0">
        <div className="flex flex-col items-start">
          <LabelLink href="/platform/#agent" className="text-ink">Let the agent work a folder</LabelLink>
          <LabelLink href="/mcp/" className="text-ink">Connect an MCP client</LabelLink>
          <LabelLink href="/platform/#questions" className="text-ink">Questions</LabelLink>
        </div>
      </RailSection>
      <RailSection id="verify" label="Verify" className="!pt-0">
        <Md html={render(section(readme, "Verify"))} />
      </RailSection>
    </PageShell>
  );
}
