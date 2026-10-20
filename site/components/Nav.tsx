import Link from "next/link";
import { KnotLockup } from "./Knot";
import { SITE } from "@/lib/site";

const LINKS = [
  { href: "/platform/", label: "Platform" },
  { href: "/mcp/", label: "MCP" },
  { href: "/install/", label: "Install" },
  { href: "/updates/", label: "Updates" },
];

export function Nav({ tone = "ink", overlay = false }: { tone?: "ink" | "white"; overlay?: boolean }) {
  const text = tone === "white" ? "text-white" : "text-ink";
  const line = tone === "white" ? "border-white/35" : "border-hair";
  const item = "text-[12px] font-semibold uppercase tracking-[0.14em] py-2 hover:underline underline-offset-4 decoration-1";
  return (
    <header className={`${overlay ? "absolute inset-x-0 top-0" : "relative"} z-20 ${text}`}>
      <div className={`wrap flex h-[72px] items-center justify-between border-b ${line} lg:h-[88px]`}>
        <Link href="/" className="-my-2 py-2" aria-label="Angular AI, home">
          <KnotLockup id={`nav-${tone}`} tone={tone} className="h-[28px] w-auto lg:h-[34px]" />
        </Link>
        <nav aria-label="Main" className="hidden md:block">
          <ul className="flex items-center gap-8">
            {LINKS.map((l) => (
              <li key={l.href}><Link href={l.href} className={item}>{l.label}</Link></li>
            ))}
            <li><a href={SITE.repo} className={item}>GitHub</a></li>
          </ul>
        </nav>
        <details className="group relative md:hidden">
          <summary className={`${item} list-none cursor-pointer select-none [&::-webkit-details-marker]:hidden`}>
            <span className="group-open:hidden">Menu</span>
            <span className="hidden group-open:inline">Close</span>
          </summary>
          <nav aria-label="Menu" className="absolute right-0 top-full mt-3 w-56 rounded-md bg-white p-2 text-ink shadow-[0_20px_50px_-20px_rgb(11_11_15/0.5)] ring-1 ring-black/5">
            <ul>
              {LINKS.map((l) => (
                <li key={l.href}><Link href={l.href} className="block rounded px-3 py-3 text-[15px] font-medium hover:bg-mist">{l.label}</Link></li>
              ))}
              <li><a href={SITE.repo} className="block rounded px-3 py-3 text-[15px] font-medium hover:bg-mist">GitHub</a></li>
            </ul>
          </nav>
        </details>
      </div>
    </header>
  );
}
