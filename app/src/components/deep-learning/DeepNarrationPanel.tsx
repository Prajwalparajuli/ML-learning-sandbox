import type { DeepNarrationItem } from '@/data/deep-learning/types';

interface DeepNarrationPanelProps {
  title: string;
  narration: DeepNarrationItem | null;
}

export function DeepNarrationPanel({ title, narration }: DeepNarrationPanelProps) {
  if (!narration) {
    return (
      <section className="deep-panel deep-panel-secondary">
        <h4 className="deep-section-title">{title}</h4>
        <p className="deep-section-subtitle">Step through the model to activate guided narration.</p>
      </section>
    );
  }

  return (
    <section className="deep-panel deep-panel-secondary">
      <h4 className="deep-section-title">{title}</h4>
      <div className="deep-narration-grid">
        <p><strong>What changed:</strong> {narration.what_changed}</p>
        <p><strong>Why it changed:</strong> {narration.why}</p>
        <p><strong>Try next:</strong> {narration.try_next}</p>
        <p><strong>Misconception to avoid:</strong> {narration.misconception}</p>
      </div>
      {narration.predict_prompt && (
        <div className="deep-predict-block">
          <p><strong>Predict first:</strong> {narration.predict_prompt}</p>
          <p className="deep-feature-copy"><strong>Reveal:</strong> {narration.reveal_text ?? 'Observe the updated state and compare with your prediction.'}</p>
        </div>
      )}
    </section>
  );
}
