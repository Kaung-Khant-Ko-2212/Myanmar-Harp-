import { Music2 } from 'lucide-react';

const navItems = [
  { href: '/#home', label: 'Home' },
  { href: '/#workflow', label: 'Workflow' },
  { href: '/#about', label: 'About' },
  { href: '/#contact', label: 'Contact' },
];

const SiteNav = () => {
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-white/10 bg-background/95 backdrop-blur-xl">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 md:px-6">
        <a href="/#home" className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-white/15 bg-white/10">
            <Music2 className="h-5 w-5 text-cyan-300" />
          </span>
          <span className="truncate font-heading text-lg font-bold text-white">
            Myanmar Harp
          </span>
        </a>

        <div className="hidden items-center gap-1 md:flex">
          {navItems.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="rounded-lg px-3 py-2 text-sm text-white/70 transition-colors hover:bg-white/10 hover:text-white"
            >
              {item.label}
            </a>
          ))}
        </div>

        <a
          href="/analyzer"
          className="rounded-lg border border-cyan-300/35 bg-cyan-300/12 px-4 py-2 text-sm font-medium text-cyan-100 transition-colors hover:bg-cyan-300/20"
        >
          Analyze
        </a>
      </nav>
    </header>
  );
};

export default SiteNav;
