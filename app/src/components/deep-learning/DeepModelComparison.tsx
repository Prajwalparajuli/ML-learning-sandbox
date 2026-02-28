import comparisonData from '@/data/deep-learning/model_comparison.json';
import type { ModelComparisonRecord } from '@/data/deep-learning/types';

const models = comparisonData as ModelComparisonRecord[];

export function DeepModelComparison() {
  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Model Comparison Cards</h3>
        <p className="deep-section-subtitle">Precomputed tradeoffs across model size, accuracy, and latency.</p>
      </div>
      <div className="deep-quiz-grid">
        {models.map((model) => (
          <article key={model.id} className="deep-quiz-card">
            <p className="deep-quiz-question">{model.label}</p>
            <p className="deep-quiz-expl">Params: {model.params_millions.toFixed(2)}M</p>
            <p className="deep-quiz-expl">Top-1 Accuracy: {(model.top1_accuracy * 100).toFixed(1)}%</p>
            <p className="deep-quiz-expl">Latency: {model.latency_ms.toFixed(1)} ms</p>
            <p className="deep-quiz-answer">{model.note}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
