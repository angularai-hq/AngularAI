import type { Metadata, Viewport } from "next";
import { Inter, Newsreader } from "next/font/google";
import "./globals.css";
import { SITE } from "@/lib/site";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const newsreader = Newsreader({ subsets: ["latin"], variable: "--font-newsreader", display: "swap", weight: "400", style: "normal" });

export const metadata: Metadata = {
  metadataBase: new URL(`${SITE.url}/`),
  title: { default: "Angular AI: a lightweight Thameleon", template: "%s · Angular AI" },
  description: SITE.description,
  applicationName: "Angular AI",
  authors: [{ name: "Teodor Walter Vido" }],
  openGraph: {
    type: "website",
    siteName: "Angular AI",
    title: "Angular AI: a lightweight Thameleon",
    description: SITE.description,
    images: [{ url: `${SITE.url}/og.png`, width: 1200, height: 630, alt: "Angular AI on a teal, violet, yellow and pink gradient." }],
  },
  twitter: { card: "summary_large_image", images: [`${SITE.url}/og.png`] },
  robots: { index: true, follow: true },
};

// Scroll reveals. Marks <html data-motion> only when motion is allowed, so without JS or with
// reduced motion nothing is ever hidden. Uses data-in (not a class) so React hydration is untouched.
const MOTION = `(function(){try{var d=document.documentElement;if(!window.IntersectionObserver||matchMedia("(prefers-reduced-motion: reduce)").matches)return;d.setAttribute("data-motion","");var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.setAttribute("data-in","");io.unobserve(e.target)}})},{rootMargin:"0px 0px -6% 0px",threshold:0});var seen=new WeakSet();function scan(){document.querySelectorAll("[data-reveal],.type-on").forEach(function(el){if(!seen.has(el)){seen.add(el);io.observe(el)}})}document.addEventListener("DOMContentLoaded",function(){scan();new MutationObserver(scan).observe(document.body,{childList:true,subtree:true})})}catch(e){}})();`;

export const viewport: Viewport = { themeColor: "#ffffff", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${newsreader.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: MOTION }} />
      </head>
      <body>
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-white focus:px-4 focus:py-2 focus:text-ink focus:shadow">
          Skip to content
        </a>
        {children}
      </body>
    </html>
  );
}
