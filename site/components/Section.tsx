/** Reference layout: small label in a 200px rail, content in the main column. */
export function RailSection({ id, label, children, className = "", reveal = "" }: { id?: string; label: string; children: React.ReactNode; className?: string; reveal?: "" | "fade" }) {
  const hid = id ? `${id}-label` : undefined;
  return (
    <section id={id} aria-labelledby={hid} className={`wrap py-20 lg:py-28 ${className}`}>
      <div className="rail">
        <h2 id={hid} data-reveal="fade" className="pt-1 text-[14px] font-medium text-ink">{label}</h2>
        <div className="min-w-0" data-reveal={reveal} style={{ "--d": 80 } as React.CSSProperties}>{children}</div>
      </div>
    </section>
  );
}
