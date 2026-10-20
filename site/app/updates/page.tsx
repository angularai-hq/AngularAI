import type { Metadata } from "next";
import { PageShell } from "@/components/PageShell";
import { RailSection } from "@/components/Section";
import { UpdateList } from "@/components/UpdateList";
import { LabelLink } from "@/components/Chevron";
import { UPDATES } from "@/lib/updates";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Updates",
  description: "Every change to Angular AI, dated, from the merged pull requests and commits on main.",
  alternates: { canonical: `${SITE.url}/updates/` },
};

export default function Updates() {
  return (
    <PageShell label="Updates" title="Every change, dated." lede={<p>Each item is a merged pull request or a commit on main. Dates are in Central European Summer Time.</p>}>
      <RailSection id="all" label="All updates" className="!pt-4">
        <UpdateList items={UPDATES} />
        <LabelLink href={`${SITE.repo}/commits/main`} className="mt-4 text-ink">Full history on GitHub</LabelLink>
      </RailSection>
    </PageShell>
  );
}
