import { useEffect, useRef } from 'react';

const AnimatedBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
      hue: number;
    }> = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const createParticles = () => {
      particles = [];
      const count = Math.floor((canvas.width * canvas.height) / 15000);
      
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          size: Math.random() * 2 + 0.5,
          opacity: Math.random() * 0.5 + 0.1,
          hue: Math.random() > 0.5 ? 187 : 258, // cyan or purple
        });
      }
    };

    const drawGradientOrbs = (time: number) => {
      // Large floating orbs
      const orbs = [
        { x: canvas.width * 0.2, y: canvas.height * 0.3, radius: 200, hue: 187 },
        { x: canvas.width * 0.8, y: canvas.height * 0.7, radius: 250, hue: 258 },
        { x: canvas.width * 0.5, y: canvas.height * 0.5, radius: 180, hue: 43 },
      ];

      orbs.forEach((orb, i) => {
        const offsetX = Math.sin(time * 0.001 + i) * 30;
        const offsetY = Math.cos(time * 0.0008 + i * 2) * 20;
        
        const gradient = ctx.createRadialGradient(
          orb.x + offsetX, orb.y + offsetY, 0,
          orb.x + offsetX, orb.y + offsetY, orb.radius
        );
        
        gradient.addColorStop(0, `hsla(${orb.hue}, 80%, 50%, 0.08)`);
        gradient.addColorStop(0.5, `hsla(${orb.hue}, 80%, 50%, 0.03)`);
        gradient.addColorStop(1, 'transparent');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      });
    };

    const drawParticles = () => {
      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around edges
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, 60%, ${p.opacity})`;
        ctx.fill();
      });

      // Draw connections
      particles.forEach((p1, i) => {
        particles.slice(i + 1).forEach(p2 => {
          const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `hsla(187, 80%, 50%, ${(1 - dist / 120) * 0.1})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });
    };

    const animate = (time: number) => {
      ctx.fillStyle = 'hsl(220, 20%, 7%)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      drawGradientOrbs(time);
      drawParticles();
      
      animationId = requestAnimationFrame(animate);
    };

    resize();
    createParticles();
    animate(0);

    window.addEventListener('resize', () => {
      resize();
      createParticles();
    });

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10"
      style={{ background: 'hsl(220, 20%, 7%)' }}
    />
  );
};

export default AnimatedBackground;
