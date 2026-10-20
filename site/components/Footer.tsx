import Link from "next/link";
import { Mark } from "./Logo";
import { SITE, asset } from "@/lib/site";

export function Footer() {
  const a = "hover:text-ink hover:underline underline-offset-4";
  return (
    <footer className="wrap mt-8">
      <div className="flex flex-col gap-5 border-t border-hair py-8 text-[13px] text-quiet sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <Mark className="h-5 w-auto" />
          <span>Angular AI · MIT licence · Teodor Walter Vido · 2026</span>
        </div>
        <nav aria-label="Footer">
          <ul className="flex flex-wrap gap-x-5 gap-y-2">
            <li><Link className={a} href="/platform/">Platform</Link></li>
            <li><Link className={a} href="/mcp/">MCP</Link></li>
            <li><Link className={a} href="/install/">Install</Link></li>
            <li><Link className={a} href="/updates/">Updates</Link></li>
            <li><a className={a} href={asset("/angularai.pdf")}>Paper</a></li>
            <li><a className={a} href={SITE.repo}>GitHub</a></li>
          </ul>
        </nav>
      </div>
    </footer>
  );
}
