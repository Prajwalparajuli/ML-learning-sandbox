import type { ReactNode } from 'react';
import { Layers2 } from 'lucide-react';
import { deepLearningContent } from '@/content/deepLearningContent';

interface DeepLearningHeroProps {
  children?: ReactNode;
}

export function DeepLearningHero({ children }: DeepLearningHeroProps) {
  return (
    <section className="deep-panel deep-hero">
      <p className="deep-eyebrow">{deepLearningContent.hero.eyebrow}</p>
      <h2 className="deep-title">{deepLearningContent.hero.title}</h2>
      <p className="deep-subtitle">Client-side deep-learning studio focused on clear, step-based real inference learning.</p>
      <div className="deep-pill-row">
        <span className="deep-pill">
          <Layers2 className="w-3.5 h-3.5" />
          <span>Browser inference</span>
        </span>
        <span className="deep-pill">No server compute</span>
        <span className="deep-pill">Step-by-step visuals</span>
      </div>
      {children && <div className="deep-hero-toolbar">{children}</div>}
    </section>
  );
}
