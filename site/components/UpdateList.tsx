import { fmtDate, type Update } from "@/lib/updates";

export function UpdateList({ items }: { items: Update[] }) {
  return (
    <ol className="border-t border-hair">
      {items.map((u) => (
        <li key={u.href} className="border-b border-hair">
          <a href={u.href} className="group grid gap-1 py-4 sm:grid-cols-[200px_minmax(0,1fr)_auto] sm:items-baseline sm:gap-12 sm:py-[18px] lg:grid-cols-[200px_minmax(0,1fr)_auto]">
            <time dateTime={u.date} className="date text-quiet">{fmtDate(u.date)}</time>
            <span className="text-[15px] font-medium text-ink group-hover:underline underline-offset-4 decoration-1">{u.title}</span>
            <span className="hidden text-[13px] text-quiet sm:inline">{u.ref}</span>
          </a>
        </li>
      ))}
    </ol>
  );
}
