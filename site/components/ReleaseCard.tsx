import Link from "next/link";
import { asset } from "@/lib/site";
import { fmtDate, type Release } from "@/lib/updates";

export function ReleaseCard({ r, eager = false, first = false, i = 0 }: { r: Release; eager?: boolean; first?: boolean; i?: number }) {
  return (
    <Link href={r.href} style={{ "--d": 260 + i * 90 } as React.CSSProperties} className="card rise group block w-[78vw] max-w-[320px] shrink-0 snap-start bg-white p-3 text-ink sm:w-auto sm:max-w-none">
      <span className="flex min-h-[48px] flex-col justify-start px-1 pb-3 pt-1">
        <span className="text-[15px] font-semibold leading-[1.3]">{r.title}</span>
        <span className="date mt-1.5 text-quiet">{fmtDate(r.date)} <span className="chev" aria-hidden="true">›</span></span>
      </span>
      <span className="art block">
      <img
        src={asset(`/art/${r.art}-420.webp`)}
        srcSet={`${asset(`/art/${r.art}-420.webp`)} 420w, ${asset(`/art/${r.art}-800.webp`)} 800w`}
        sizes="(min-width: 1024px) 22vw, (min-width: 640px) 45vw, 78vw"
        width={420}
        height={560}
        alt={r.alt}
        loading={eager ? "eager" : "lazy"}
        fetchPriority={first ? "high" : undefined}
        decoding="async"
        className="aspect-[3/4] h-auto w-full object-cover"
      />
      </span>
      <span className="sr-only">. {r.summary}</span>
    </Link>
  );
}
