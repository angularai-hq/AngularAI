import type { MetadataRoute } from "next";
import { SITE } from "@/lib/site";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const pages = ["/", "/platform/", "/mcp/", "/install/", "/updates/"];
  return pages.map((p) => ({ url: `${SITE.url}${p}`, lastModified: "2026-11-10", changeFrequency: "weekly", priority: p === "/" ? 1 : 0.7 }));
}
