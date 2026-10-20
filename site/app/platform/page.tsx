import type { Metadata } from "next";
import { PageShell, Md } from "@/components/PageShell";
import { RailSection } from "@/components/Section";
import { LabelLink } from "@/components/Chevron";
import { LAYERS } from "@/lib/layers";
import { readRepo, render, section } from "@/lib/md";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Platform",
  description: "The cortex, the membrane, the value lock, the agent and the MCP connector: how Angular AI works on top of the model you already use.",
  alternates: { canonical: `${SITE.url}/platform/` },
};

const L = Object.fromEntries(LAYERS.map((l) => [l.id, l]));

export default function Platform() {
  const readme = readRepo("README.md");
  const work = render(section(readme, "Work"));
  const talk = render(section(readme, "Talk"));
  const serve = render(section(readme, "Serve"));
  const questions = render(section(readme, "Questions"));
  return (
    <PageShell
      label="Platform"
      title="A lightweight Thameleon."
      lede={
        <p>
          Angular AI brings Thameleon&rsquo;s brain-like CLASP architecture to software. It is a platform on top of a model you already call: OpenAI, Anthropic, or a compatible endpoint. There is no custom model and no training. The vendor&rsquo;s weights do not change.
        </p>
      }
    >
      <nav aria-label="On this page" className="wrap">
        <div className="rail">
          <div />
          <ul className="flex flex-wrap gap-x-6 gap-y-1 border-y border-hair py-3">
            {[["cortex", "Cortex"], ["membrane", "Membrane"], ["recall", "Recall"], ["lock", "Value lock"], ["agent", "Agent"], ["talk", "Cortexes that talk"], ["mcp", "MCP"], ["questions", "Questions"]].map(([id, t]) => (
              <li key={id}><a href={`#${id}`} className="label-link text-ink"><span>{t}</span></a></li>
            ))}
          </ul>
        </div>
      </nav>

      <RailSection id="cortex" label="Cortex">
        <div className="md">
          <h3 className="!mt-0">{L.cortex.line}</h3>
          <p>{L.cortex.body}</p>
          <p>The cortex is <code>~/.angularai/default</code>, readable only by you. The key stays in the environment. It is not a field in the cortex.</p>
        </div>
      </RailSection>

      <RailSection id="membrane" label="Membrane" className="!pt-0">
        <div className="md">
          <h3 className="!mt-0">{L.membrane.line}</h3>
          <p>{L.membrane.body}</p>
          <p>Four ASCII characters to one token. A wide character (CJK, kana, Hangul, fullwidth) is one; one above U+FFFF, like most emoji, is two. Any other character is half. Rounded up. Angular AI enforces that estimate. It is not the provider&rsquo;s tokenizer.</p>
        </div>
      </RailSection>

      <RailSection id="recall" label="Recall" className="!pt-0">
        <div className="md">
          <h3 className="!mt-0">The rest can return when a question shares its words.</h3>
          <table>
            <thead><tr><th>one pass</th><th>text</th></tr></thead>
            <tbody>
              <tr><td>asked</td><td>What color is the notebook?</td></tr>
              <tr><td>found</td><td>The notebook is blue.</td></tr>
              <tr><td>missed</td><td>What shade is it?</td></tr>
            </tbody>
          </table>
          <p>Lexical attention by default. <code>ANGULARAI_EMBEDDINGS</code> adds recall by meaning. With embeddings on, &ldquo;When is lunch?&rdquo; finds &ldquo;Catering is at noon.&rdquo; by meaning. A trace found by meaning still enters only if it fits the budget.</p>
        </div>
      </RailSection>

      <RailSection id="lock" label="Value lock" className="!pt-0">
        <div className="md">
          <h3 className="!mt-0">{L.lock.line}</h3>
          <p>A secret in the input is refused before the call, and before any line is written. A phrase you forbid, or a secret, in the reply or in the tool it asks for, holds it. Case, width, spacing, zero-width characters, and Cyrillic or Greek letters drawn like Latin ones do not hide a phrase. You see one sentence: <code>Value lock held this reply.</code> The held text is not stored.</p>
        </div>
      </RailSection>

      <RailSection id="agent" label="Agent" className="!pt-0">
        <div className="md"><h3 className="!mt-0">{L.agent.line}</h3></div>
        <Md html={work} className="mt-4" />
      </RailSection>

      <RailSection id="talk" label="Cortexes that talk" className="!pt-0">
        <div className="md"><h3 className="!mt-0">Two cortexes take turns. Only a message crosses.</h3></div>
        <Md html={talk} className="mt-4" />
      </RailSection>

      <RailSection id="mcp" label="MCP" className="!pt-0">
        <div className="md"><h3 className="!mt-0">{L.mcp.line}</h3></div>
        <Md html={serve} className="mt-4" />
        <LabelLink href="/mcp/" className="mt-4 text-ink">MCP guide</LabelLink>
      </RailSection>

      <RailSection id="questions" label="Questions" className="!pt-0">
        <Md html={questions} />
      </RailSection>
    </PageShell>
  );
}
