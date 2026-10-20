import type { MetadataRoute } from "next";
import { SITE } from "@/lib/site";

export const dynamic = "force-static";

// GitHub Pages serves this at /AngularAI/robots.txt; crawlers read only the host root,
// so the sitemap is also linked from every page head. Kept for completeness.
export default function robots(): MetadataRoute.Robots {
  return { rules: [{ userAgent: "*", allow: "/" }], sitemap: `${SITE.url}/sitemap.xml` };
}
