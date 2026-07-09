import AnimatedBackground from '@/components/AnimatedBackground';
import Hero from '@/components/Hero';
import SiteNav from '@/components/SiteNav';
import SiteSections from '@/components/SiteSections';
import { useEffect } from 'react';

const Home = () => {
  useEffect(() => {
    const scrollToHash = () => {
      const targetId = window.location.hash.replace('#', '');
      if (!targetId) {
        return;
      }

      window.requestAnimationFrame(() => {
        const target = document.getElementById(targetId);
        target?.scrollIntoView({ block: 'start' });
      });
    };

    scrollToHash();
    window.addEventListener('hashchange', scrollToHash);

    return () => {
      window.removeEventListener('hashchange', scrollToHash);
    };
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden">
      <AnimatedBackground />
      <SiteNav />

      <div className="relative z-10 pb-16 md:pb-24">
        <Hero />
        <SiteSections />
      </div>

      <footer className="relative z-10 border-t border-white/10 px-4 py-8">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 text-xs text-muted-foreground/65 md:flex-row md:items-center md:justify-between">
          <p>Myanmar Harp String Detection | Computer Vision Research Project</p>
          <div className="flex gap-4">
            <a href="/#home" className="transition-colors hover:text-white">Home</a>
            <a href="/#workflow" className="transition-colors hover:text-white">Workflow</a>
            <a href="/#about" className="transition-colors hover:text-white">About</a>
            <a href="/#contact" className="transition-colors hover:text-white">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;
