import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import { Footer } from "@/components/Footer";
import { LabelLink } from "@/components/Chevron";
import { ReleaseCard } from "@/components/ReleaseCard";
import { UpdateList } from "@/components/UpdateList";
import { RailSection } from "@/components/Section";
import { CodeBlock } from "@/components/CodeBlock";
import { Film } from "@/components/Film";
import { Mark } from "@/components/Logo";
import { RELEASES, UPDATES } from "@/lib/updates";
import { LAYERS } from "@/lib/layers";
import { SITE, asset } from "@/lib/site";

export const metadata: Metadata = {
  title: { absolute: "Angular AI: a lightweight Thameleon" },
  alternates: { canonical: `${SITE.url}/` },
};

export default function Home() {
  return (
    <>
      <Nav tone="white" overlay />
      <main id="main">
          <section aria-labelledby="hero-title" className="relative isolate overflow-hidden pt-[72px] text-white lg:pt-[88px]">
            <div className="mesh -z-10" aria-hidden="true">
              <i className="b1" /><i className="b2" /><i className="b5" /><i className="b3" /><i className="b4" /><i className="b6" />
            </div>
            <div className="wrap pb-14 pt-12 lg:pb-20 lg:pt-16">
              <div className="rail">
                <p className="pt-2 text-[14px] font-medium">Platform</p>
                <div>
                  <h1 id="hero-title" className="max-w-[22em] font-serif text-[29px] leading-[1.2] tracking-[-0.01em] sm:text-[34px] lg:text-[40px]">
                    Angular AI brings Thameleon&rsquo;s brain-like architecture to the model you already use.
                  </h1>
                  <LabelLink href="/platform/" className="mt-5 text-white">How it works</LabelLink>
                </div>
              </div>
            </div>
            <div className="wrap">
              <div className="border-t border-white/35 pt-6 lg:pt-8">
                <h2 className="text-[14px] font-medium">Recent releases</h2>
              </div>
            </div>
            <div className="wrap mt-5 lg:mt-6">
              <div className="snap-row -mx-[var(--gutter)] flex snap-x snap-mandatory scroll-px-[var(--gutter)] gap-4 overflow-x-auto px-[var(--gutter)] pb-2 sm:mx-0 sm:grid sm:grid-cols-2 sm:gap-6 sm:overflow-visible sm:px-0 lg:grid-cols-4 lg:gap-8">
                {RELEASES.map((r, i) => <ReleaseCard key={r.href} r={r} eager={i < 2} first={i === 0} />)}
              </div>
            </div>
            <div className="wrap flex justify-center pb-10 pt-8 lg:pb-14 lg:pt-12">
              <LabelLink href="/updates/" className="text-ink">View all updates</LabelLink>
            </div>
          </section>
        <section aria-label="Mission" className="wrap py-24 lg:py-36">
          <div className="rail">
            <div aria-hidden="true" />
            <div>
              <p className="max-w-[30em] font-serif text-[24px] leading-[1.42] text-ink sm:text-[28px] lg:text-[32px]">
                Angular AI is not a lightweight tool. It is a lightweight Thameleon. It brings Thameleon&rsquo;s brain-like CLASP architecture to software, as a platform on top of any existing model. There is no custom model and no training. It is free and open source.
              </p>
              <LabelLink href="/platform/" className="mt-6 text-ink">About the platform</LabelLink>
            </div>
          </div>
        </section>

        <RailSection id="layers" label="The layers" className="!pt-0">
          <ol className="grid gap-x-12 gap-y-10 sm:grid-cols-2 xl:grid-cols-3">
            {LAYERS.map((l, i) => (
              <li key={l.id} className="border-t border-hair pt-5">
                <p className="date text-quiet">0{i + 1}</p>
                <h3 className="mt-3 font-serif text-[24px] leading-tight text-ink">{l.name}</h3>
                <p className="mt-2 text-[16px] leading-[1.6]">{l.line}</p>
                <LabelLink href={l.id === "mcp" ? "/mcp/" : `/platform/#${l.id}`} className="mt-2 text-ink">More on the {l.id === "mcp" ? "MCP connector" : l.name.toLowerCase()}</LabelLink>
              </li>
            ))}
          </ol>
        </RailSection>

        <RailSection id="film" label="In twelve seconds" className="!pt-4">
          <Film />
          <p className="mt-3 text-[13px] text-quiet">The film has no sound. The memories, numbers and replies in it are illustrative.</p>
        </RailSection>

        <RailSection id="install" label="Install">
          <div className="max-w-[760px]">
            <CodeBlock code={SITE.install} />
            <p className="mt-4 text-[15px]">Then wire the API you already have. The key stays in the environment.</p>
            <LabelLink href="/install/" className="mt-2 text-ink">Install guide</LabelLink>
          </div>
        </RailSection>

        <RailSection id="mcp" label="MCP" className="!pt-4">
          <div className="max-w-[760px]">
            <p className="mb-5 text-[17px] leading-[1.6]">
              The cortex as an MCP server. Five tools: <code>remember</code>, <code>recall</code>, <code>forget</code>, <code>status</code>, <code>cortexes</code>. Stdio by default. <code>--http</code> serves streamable HTTP on 127.0.0.1.
            </p>
            <CodeBlock code={SITE.mcpRun} />
            <p className="mt-4 text-[15px]">
              Claude Desktop, Claude Code, Codex, Cursor, or any MCP client.{" "}
              <a className="inline" href="https://github.com/angularai-hq/AngularAI/tree/main/packages/angularai-mcp#setup">Setup for each client</a>.
            </p>
            <LabelLink href="/mcp/" className="mt-2 text-ink">MCP guide</LabelLink>
          </div>
        </RailSection>

        <RailSection id="updates" label="Recent updates">
          <UpdateList items={UPDATES.slice(0, 5)} />
          <LabelLink href="/updates/" className="mt-4 text-ink">View all updates</LabelLink>
        </RailSection>

        <section aria-labelledby="join-title" className="wrap pb-20 lg:pb-28">
          <div className="grid gap-10 border-t border-hair pt-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-12 lg:pt-16">
            <div className="flex flex-col">
              <h2 id="join-title" className="text-[20px] font-semibold text-ink">Build Angular AI with us</h2>
              <p className="mt-3 max-w-[26em] text-[16px] leading-[1.6]">
                Angular AI is free and open source under the MIT licence. The code, the issues and the pull requests are on GitHub. Every pull request runs the same gates in CI before it merges.
              </p>
              <ul className="mt-6 flex flex-col items-start lg:mt-auto lg:pt-10">
                <li><LabelLink href={SITE.repo} className="text-ink">View on GitHub</LabelLink></li>
                <li><LabelLink href={`${SITE.repo}/issues`} className="text-ink">Open an issue</LabelLink></li>
                <li><LabelLink href={`${SITE.repo}/pulls`} className="text-ink">Send a pull request</LabelLink></li>
              </ul>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <img
                src={asset("/art/join-640.webp")}
                srcSet={`${asset("/art/join-640.webp")} 640w, ${asset("/art/join-1152.webp")} 1152w`}
                sizes="(min-width: 1024px) 40vw, 66vw"
                width={640}
                height={480}
                loading="lazy"
                decoding="async"
                alt="Black, white and grey blocks fitting together into a knot, with coloured blocks arriving from the edges."
                className="col-span-2 h-full w-full bg-mist object-cover"
              />
              <div className="grid gap-4">
                <div className="flex items-center justify-center bg-mist p-6"><Mark className="h-16 w-auto" /></div>
                <a href={asset("/angularai.pdf")} className="group flex flex-col justify-end bg-ink p-4 text-white">
                  <span className="date text-white/70">PDF</span>
                  <span className="mt-1 font-serif text-[20px] leading-tight group-hover:underline underline-offset-4">Read the paper</span>
                </a>
              </div>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
