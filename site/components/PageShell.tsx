import { Nav } from "./Nav";
import { Footer } from "./Footer";

export function PageShell({ label, title, lede, children }: { label: string; title: string; lede?: React.ReactNode; children: React.ReactNode }) {
  return (
    <>
      <div className="mesh-strip h-1.5 w-full" aria-hidden="true" />
      <Nav />
      <main id="main">
        <header className="wrap pb-10 pt-16 lg:pb-16 lg:pt-24">
          <div className="rail">
            <p className="pt-3 text-[14px] font-medium text-ink">{label}</p>
            <div>
              <h1 className="max-w-[18em] font-serif text-[36px] leading-[1.1] tracking-[-0.015em] text-ink sm:text-[46px] lg:text-[56px]">{title}</h1>
              {lede ? <div className="mt-6 max-w-[34em] font-serif text-[20px] leading-[1.5] text-body lg:text-[23px]">{lede}</div> : null}
            </div>
          </div>
        </header>
        {children}
      </main>
      <Footer />
    </>
  );
}

export function Md({ html, className = "" }: { html: string; className?: string }) {
  return <div className={`md ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
