"use client";
import { useEffect, useRef } from "react";
import { asset } from "@/lib/site";

/** Muted 12 s film. Loads nothing until near the viewport; never autoplays with reduced motion. */
export function Film() {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(v);
    return () => io.disconnect();
  }, []);
  return (
    <video
      ref={ref}
      className="aspect-video w-full rounded-lg border border-hair bg-white"
      muted
      loop
      playsInline
      controls
      preload="none"
      poster={asset("/media/botstyle-poster.webp")}
      width={1280}
      height={720}
      aria-label="A 12 second film: Angular AI recalls four memories from the cortex, sends what fits the budget, and blocks one API key."
    >
      <source src={asset("/media/botstyle.mp4")} type="video/mp4" />
      <source src={asset("/media/botstyle.webm")} type="video/webm" />
    </video>
  );
}
