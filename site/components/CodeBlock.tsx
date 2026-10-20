"use client";
import { useState } from "react";

/** A real code / install block: the one place monospace is used. */
export function CodeBlock({ code, label = "terminal", copy = true, typing = false }: { code: string; label?: string; copy?: boolean; typing?: boolean }) {
  const [state, setState] = useState<"idle" | "done" | "fail">("idle");
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code.endsWith("\n") ? code : `${code}\n`);
      setState("done");
    } catch {
      setState("fail");
    }
    setTimeout(() => setState("idle"), 2000);
  };
  return (
    <figure
      className={`overflow-hidden rounded-lg border border-hair bg-[#0f1016] text-[#e8e8ef] ${typing ? "type-on" : ""}`}
      style={typing ? ({ "--n": code.length, "--type-ms": `${Math.min(2200, code.length * 24)}ms` } as React.CSSProperties) : undefined}
    >
      <figcaption className="flex items-center justify-between border-b border-white/10 px-4 py-2 text-[12px] text-[#b9b9c6]">
        <span>{label}</span>
        {copy ? (
          <button type="button" onClick={onCopy} className="rounded px-2 py-1 font-semibold uppercase tracking-[0.12em] text-[11px] text-white hover:bg-white/10" aria-live="polite">
            {state === "done" ? "Copied" : state === "fail" ? "Select it" : "Copy"}
          </button>
        ) : null}
      </figcaption>
      <pre className="overflow-x-auto px-4 py-4 text-[13.5px] leading-[1.65]" tabIndex={0} aria-label={`${label} code`}>
        {typing ? (
          <code>
            <span className="prompt" aria-hidden="true">$ </span>
            <span className="typed">{code}</span>
            <span className="caret" aria-hidden="true" />
          </code>
        ) : (
          <code>{code}</code>
        )}
      </pre>
    </figure>
  );
}
