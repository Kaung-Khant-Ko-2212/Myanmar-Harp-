import { useEffect, useState } from 'react';

interface SoundWaveLoaderProps {
  status: 'analyzing' | 'detecting' | 'generating';
}

const SoundWaveLoader = ({ status }: SoundWaveLoaderProps) => {
  const [activeBars, setActiveBars] = useState<number[]>([]);
  const barCount = 12;

  useEffect(() => {
    const interval = setInterval(() => {
      const newHeights = Array.from({ length: barCount }, () => 
        Math.random() * 60 + 20
      );
      setActiveBars(newHeights);
    }, 150);

    return () => clearInterval(interval);
  }, []);

  const statusMessages = {
    analyzing: 'Analyzing audio...',
    detecting: 'Detecting patterns...',
    generating: 'Generating notes...',
  };

  const statusProgress = {
    analyzing: 33,
    detecting: 66,
    generating: 90,
  };

  return (
    <div className="flex flex-col items-center gap-8 py-8">
      {/* Sound Wave Visualization */}
      <div className="relative w-72 h-48 flex items-center justify-center">
        {/* Glow background */}
        <div className="absolute inset-0 bg-primary/20 rounded-full blur-3xl animate-pulse" />
        
        {/* Circular backdrop */}
        <div className="absolute w-40 h-40 rounded-full border border-primary/20 bg-primary/5" />
        <div className="absolute w-52 h-52 rounded-full border border-primary/10" />
        
        {/* Equalizer bars */}
        <div className="relative flex items-end justify-center gap-1.5 h-24">
          {Array.from({ length: barCount }).map((_, i) => {
            const height = activeBars[i] || 30;
            const delay = i * 0.05;
            
            return (
              <div
                key={i}
                className="w-2 rounded-full transition-all duration-150 ease-out"
                style={{
                  height: `${height}%`,
                  background: `linear-gradient(180deg, hsl(var(--primary)) 0%, hsl(var(--accent)) 100%)`,
                  boxShadow: `0 0 10px hsl(var(--primary) / 0.5), 0 0 20px hsl(var(--primary) / 0.3)`,
                  animationDelay: `${delay}s`,
                }}
              />
            );
          })}
        </div>

        {/* Floating music notes */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <svg className="absolute top-4 left-8 w-6 h-6 text-primary/40 animate-float" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
          </svg>
          <svg className="absolute top-8 right-10 w-5 h-5 text-accent/50 animate-float" style={{ animationDelay: '1s' }} viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
          </svg>
          <svg className="absolute bottom-12 right-6 w-4 h-4 text-primary/30 animate-float" style={{ animationDelay: '2s' }} viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
          </svg>
        </div>
      </div>

      {/* Status text */}
      <div className="flex flex-col items-center gap-4 w-full max-w-xs">
        <p className="text-primary text-sm tracking-wider font-medium text-glow-primary">
          {statusMessages[status]}
        </p>
        
        {/* Progress bar */}
        <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-1000 ease-out"
            style={{
              width: `${statusProgress[status]}%`,
              background: 'var(--gradient-primary)',
              boxShadow: 'var(--shadow-glow-sm)',
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default SoundWaveLoader;
