import { ArrowDown, Music, Play, Sparkles, Zap } from 'lucide-react';

const Hero = () => {
  return (
    <section id="home" className="relative isolate flex min-h-screen items-center justify-center overflow-hidden px-4 text-center">
      <video
        className="absolute inset-0 h-full w-full object-cover object-center"
        src="/intro-burmese-harp.mp4"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        aria-label="Woman playing Burmese harp"
      />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_42%,rgba(20,184,166,0.18),transparent_34%),linear-gradient(90deg,rgba(3,7,18,0.82),rgba(3,7,18,0.34)_48%,rgba(3,7,18,0.78))]" />
      <div className="absolute inset-0 bg-black/20" />
      <div className="absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-background via-background/75 to-transparent" />
      <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-background/70 to-transparent" />

      <div className="relative z-10 mx-auto max-w-6xl px-4 pb-20 pt-28 md:pt-32">
        <h1
          className="font-display text-5xl font-bold leading-[0.96] text-white drop-shadow-[0_6px_30px_rgba(0,0,0,0.65)] md:text-7xl lg:text-8xl animate-fade-in"
          style={{ animationDelay: '0.1s' }}
        >
          Myanmar Harp
          <br />
          <span className="text-white/90">String & Note Detection</span>
        </h1>

        <p
          className="mx-auto mt-6 max-w-3xl text-base leading-7 text-white/80 drop-shadow-[0_3px_18px_rgba(0,0,0,0.55)] md:text-xl animate-fade-in"
          style={{ animationDelay: '0.2s' }}
        >
          Transform traditional saung-gauk performances into musical notation using
          advanced computer vision and acoustic analysis.
        </p>

        <div
          className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row animate-fade-in"
          style={{ animationDelay: '0.25s' }}
        >
          <a
            href="/analyzer"
            className="inline-flex h-12 items-center justify-center gap-2 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(34,211,238,0.24)] transition-transform hover:-translate-y-0.5"
          >
            <Play className="h-4 w-4" />
            Analyze a Video
          </a>
          <a
            href="#workflow"
            className="inline-flex h-12 items-center justify-center gap-2 rounded-xl border border-white/20 bg-black/40 px-5 text-sm font-medium text-white/90 backdrop-blur-md transition-colors hover:bg-black/50"
          >
            View Workflow
            <ArrowDown className="h-4 w-4" />
          </a>
        </div>

        <div
          className="mt-9 flex flex-wrap justify-center gap-3 animate-fade-in"
          style={{ animationDelay: '0.3s' }}
        >
          {[
            { icon: Zap, text: 'Real-time Detection' },
            { icon: Music, text: 'Note Generation' },
            { icon: Sparkles, text: 'AI-Powered' },
          ].map((badge) => {
            const Icon = badge.icon;
            return (
              <div
                key={badge.text}
                className="flex items-center gap-2 rounded-full border border-white/15 bg-black/40 px-4 py-2 text-white/80 shadow-[0_10px_30px_rgba(0,0,0,0.28)] backdrop-blur-md transition-colors duration-300 hover:bg-black/50"
              >
                <Icon className="h-4 w-4 text-accent" />
                <span className="text-sm">{badge.text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Hero;
