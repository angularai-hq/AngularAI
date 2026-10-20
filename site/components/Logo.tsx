import { MARK_BOX, MARK_FACES, WORD_BOX, WORD_PATHS } from "@/lib/logo-data";

type Tone = "ink" | "white";

const FILLS: Record<Tone, Record<"black" | "grey" | "white", string | null>> = {
  ink: { black: "#0b0b0f", grey: "#cfcfd4", white: "#ffffff" },
  // On the gradient: solid faces white, light faces translucent, white faces open.
  white: { black: "#ffffff", grey: "rgba(255,255,255,0.55)", white: null },
};

/** The v2 knot mark, from the vector trace. */
export function Mark({ tone = "ink", className, title }: { tone?: Tone; className?: string; title?: string }) {
  const [x0, y0, x1, y1] = MARK_BOX;
  const f = FILLS[tone];
  return (
    <svg viewBox={`${x0} ${y0} ${x1 - x0} ${y1 - y0}`} className={className} role={title ? "img" : undefined} aria-hidden={title ? undefined : true}>
      {title ? <title>{title}</title> : null}
      {MARK_FACES.map((p, i) => (f[p.fill] ? <path key={i} d={p.d} fill={f[p.fill]!} /> : null))}
    </svg>
  );
}

/** Mark + traced "Angular AI" wordmark. */
export function Lockup({ tone = "ink", className }: { tone?: Tone; className?: string }) {
  const [x0, y0, , y1] = MARK_BOX;
  const [, , wx1] = WORD_BOX;
  const f = FILLS[tone];
  const word = tone === "ink" ? "#0b0b0f" : "#ffffff";
  return (
    <svg viewBox={`${x0} ${y0} ${wx1 - x0} ${y1 - y0}`} className={className} role="img" aria-label="Angular AI">
      {MARK_FACES.map((p, i) => (f[p.fill] ? <path key={i} d={p.d} fill={f[p.fill]!} /> : null))}
      {WORD_PATHS.map((d, i) => <path key={`w${i}`} d={d} fill={word} />)}
    </svg>
  );
}
