import type { MlpStepMath } from '@/data/deep-learning/types';

interface MlpOutputPanelProps {
  math: MlpStepMath;
}

export function MlpOutputPanel({ math }: MlpOutputPanelProps) {
  const predicted = math.probabilities.cat >= math.probabilities.dog ? 'Cat' : 'Dog';

  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h4 className="deep-section-title">Output Head</h4>
        <p className="deep-section-subtitle">Softmax probabilities and confidence metric.</p>
      </div>
      <div className="deep-output-bars">
        <div className="deep-prob-row">
          <span>Cat</span>
          <div className="deep-prob-track"><span style={{ width: `${(math.probabilities.cat * 100).toFixed(1)}%` }} /></div>
          <strong>{(math.probabilities.cat * 100).toFixed(1)}%</strong>
        </div>
        <div className="deep-prob-row">
          <span>Dog</span>
          <div className="deep-prob-track"><span style={{ width: `${(math.probabilities.dog * 100).toFixed(1)}%` }} /></div>
          <strong>{(math.probabilities.dog * 100).toFixed(1)}%</strong>
        </div>
      </div>
      <p className="deep-feature-copy">
        Predicted class: <strong>{predicted}</strong> | Confidence margin: <strong>{math.outputConfidenceGap.toFixed(3)}</strong>
      </p>
      <p className="deep-feature-copy">
        Confidence is a certainty estimate, not guaranteed correctness.
      </p>
    </section>
  );
}
