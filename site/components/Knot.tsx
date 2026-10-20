import { MARK_BOX, MARK_FACES, WORD_BOX, WORD_PATHS } from "@/lib/logo-data";

// The v2 knot, cut into six blocks around its centre (the same cut as the launch film and the CLI boot).
// Each block flies in from its own direction and snaps into place with a small overshoot; then the
// wordmark letters rise in. Pure CSS (see .knot-* in globals.css): it runs without JS, and with
// prefers-reduced-motion the final frame is shown at once.
const [x0, y0, x1, y1] = MARK_BOX;
const cx = (x0 + x1) / 2;
const cy = (y0 + y1) / 2;
const W = x1 - x0;
const ORDER = [0, 3, 1, 4, 2, 5]; // film order: alternate sides
const R = W * 3;

type Tone = "ink" | "white";
const FILLS: Record<Tone, Record<"black" | "grey" | "white", string | null>> = {
  ink: { black: "#0b0b0f", grey: "#cfcfd4", white: "#ffffff" },
  white: { black: "#ffffff", grey: "rgba(255,255,255,0.55)", white: null },
};

function Blocks({ id, tone, travel }: { id: string; tone: Tone; travel: number }) {
  const f = FILLS[tone];
  return (
    <>
      <defs>
        {[0, 1, 2, 3, 4, 5].map((k) => {
          const c = ((-90 + 60 * k) * Math.PI) / 180;
          const a0 = c - Math.PI / 6 - 0.002;
          const a1 = c + Math.PI / 6 + 0.002;
          const pts = `${cx},${cy} ${cx + R * Math.cos(a0)},${cy + R * Math.sin(a0)} ${cx + R * Math.cos(a1)},${cy + R * Math.sin(a1)}`;
          return (
            <clipPath key={k} id={`${id}-w${k}`}>
              <polygon points={pts} />
            </clipPath>
          );
        })}
      </defs>
      <g className="knot-mark">
        {[0, 1, 2, 3, 4, 5].map((k) => {
          const c = ((-90 + 60 * k) * Math.PI) / 180;
          const style = {
            "--dx": `${(Math.cos(c) * W * travel).toFixed(1)}px`,
            "--dy": `${(Math.sin(c) * W * travel).toFixed(1)}px`,
            "--i": ORDER.indexOf(k),
          } as React.CSSProperties;
          return (
            <g key={k} clipPath={`url(#${id}-w${k})`}>
              <g className="knot-b" style={style}>
                {MARK_FACES.map((p, i) => (f[p.fill] ? <path key={i} d={p.d} fill={f[p.fill]!} /> : null))}
              </g>
            </g>
          );
        })}
      </g>
    </>
  );
}

/** Animated lockup (mark + traced "Angular AI" wordmark). */
export function KnotLockup({ id, tone = "ink", className }: { id: string; tone?: Tone; className?: string }) {
  const [, , wx1] = WORD_BOX;
  const word = tone === "ink" ? "#0b0b0f" : "#ffffff";
  const firstX = (d: string) => Number(/M\s*([\d.]+)/.exec(d)?.[1] ?? 0);
  const sorted = WORD_PATHS.map((d) => ({ d, x: firstX(d) }))
    .sort((a, b) => a.x - b.x)
    .map((w, i) => ({ d: w.d, i }));
  return (
    <svg viewBox={`${x0} ${y0} ${wx1 - x0} ${y1 - y0}`} className={`knot ${className ?? ""}`} role="img" aria-label="Angular AI" style={{ overflow: "visible" }}>
      <Blocks id={id} tone={tone} travel={0.55} />
      <g>
        {sorted.map(({ d, i }) => (
          <path key={i} d={d} fill={word} className="knot-l" style={{ "--l": i } as React.CSSProperties} />
        ))}
      </g>
    </svg>
  );
}

/** Large decorative mark. */
export function KnotMark({ id, tone = "white", className }: { id: string; tone?: Tone; className?: string }) {
  return (
    <svg viewBox={`${x0} ${y0} ${W} ${y1 - y0}`} className={`knot knot-lg ${className ?? ""}`} aria-hidden="true" style={{ overflow: "visible" }}>
      <Blocks id={id} tone={tone} travel={0.9} />
    </svg>
  );
}
