import Link from "next/link";

export function Chevron({ className }: { className?: string }) {
  return (
    <svg width="7" height="10" viewBox="0 0 7 10" fill="none" aria-hidden="true" className={className}>
      <path d="M1.5 1l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** Small uppercase link label with a chevron, as in the reference. */
export function LabelLink({ href, children, className = "" }: { href: string; children: React.ReactNode; className?: string }) {
  const external = href.startsWith("http");
  const inner = (
    <>
      <span>{children}</span>
      <Chevron />
    </>
  );
  return external ? (
    <a href={href} className={`label-link ${className}`}>{inner}</a>
  ) : (
    <Link href={href} className={`label-link ${className}`}>{inner}</Link>
  );
}
