import type { MetadataRoute } from "next";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Angular AI",
    short_name: "Angular AI",
    description: "A lightweight Thameleon on top of the model you already use.",
    start_url: "/AngularAI/",
    scope: "/AngularAI/",
    display: "browser",
    background_color: "#ffffff",
    theme_color: "#ffffff",
    icons: [
      { src: "/AngularAI/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/AngularAI/icon-512.png", sizes: "512x512", type: "image/png" },
      { src: "/AngularAI/icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
