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

export const viewport: Viewport = { themeColor: "#ffffff", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${newsreader.variable}`}>
      <body>
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-white focus:px-4 focus:py-2 focus:text-ink focus:shadow">
          Skip to content
        </a>
        {children}
      </body>
    </html>
  );
}
