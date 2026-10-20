import type { Metadata } from "next";
import { PageShell } from "@/components/PageShell";
import { LabelLink } from "@/components/Chevron";

export const metadata: Metadata = { title: "Not found", robots: { index: false } };

export default function NotFound() {
  return (
    <PageShell label="404" title="This page is not here." lede={<p>The address may be old, or mistyped. The cortex is still on disk.</p>}>
      <div className="wrap pb-28">
        <div className="rail">
          <div />
          <div className="flex flex-col items-start">
            <LabelLink href="/" className="text-ink">Home</LabelLink>
            <LabelLink href="/install/" className="text-ink">Install</LabelLink>
            <LabelLink href="/updates/" className="text-ink">Updates</LabelLink>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
