import { useMemo, useState } from 'react';
import { InlineMath } from 'react-katex';
import { runCnnInference } from '@/lib/deepLearning/cnnTrace';
import { runMlpInference } from '@/lib/deepLearning/mlpTrace';
import type { MnistCnnModel, MnistMlpModel } from '@/lib/deepLearning/modelLoader';

interface DeepEvaluationPanelProps {
  modelKind: 'mnist_mlp' | 'mnist_cnn';
  model: MnistMlpModel | MnistCnnModel | null;
  image: number[][];
}

interface ShiftSample {
  dr: number;
  dc: number;
  predictedClass: number;
  confidence: number;
}

const shiftImage = (image: number[][], dr: number, dc: number): number[][] => {
  const rows = image.length;
  const cols = image[0]?.length ?? 0;
  const out = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const rr = r + dr;
      const cc = c + dc;
      if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
        out[rr][cc] = image[r][c];
      }
    }
  }
  return out;
};

const normalizedEntropy = (probabilities: number[]): number => {
  const n = Math.max(1, probabilities.length);
  const entropy = -probabilities.reduce((sum, p) => {
    if (p <= 1e-12) return sum;
    return sum + (p * Math.log(p));
  }, 0);
  return entropy / Math.log(n);
};

const topTwoGap = (probabilities: number[]): number => {
  const sorted = probabilities.slice().sort((a, b) => b - a);
  return (sorted[0] ?? 0) - (sorted[1] ?? 0);
};

export function DeepEvaluationPanel({ modelKind, model, image }: DeepEvaluationPanelProps) {
  const [shiftRadius, setShiftRadius] = useState(1);
  const hasSignal = useMemo(() => image.some((row) => row.some((value) => value > 0.06)), [image]);

  const analysis = useMemo(() => {
    if (!model || !hasSignal) return null;

    const runSnapshot = (candidate: number[][]) => (
      modelKind === 'mnist_mlp'
        ? runMlpInference(candidate, model as MnistMlpModel).snapshot
        : runCnnInference(candidate, model as MnistCnnModel).snapshot
    );

    const base = runSnapshot(image);
    const samples: ShiftSample[] = [];
    for (let dr = -shiftRadius; dr <= shiftRadius; dr += 1) {
      for (let dc = -shiftRadius; dc <= shiftRadius; dc += 1) {
        const shifted = shiftImage(image, dr, dc);
        const snapshot = runSnapshot(shifted);
        const confidence = snapshot.probabilities[snapshot.predictedClass] ?? 0;
        samples.push({
          dr,
          dc,
          predictedClass: snapshot.predictedClass,
          confidence,
        });
      }
    }

    const sameAsBase = samples.filter((sample) => sample.predictedClass === base.predictedClass).length;
    const stability = sameAsBase / Math.max(1, samples.length);
    const confidenceValues = samples.map((sample) => sample.confidence);
    const confidenceMean = confidenceValues.reduce((sum, value) => sum + value, 0) / Math.max(1, confidenceValues.length);
    const confidenceMin = Math.min(...confidenceValues);
    const confidenceMax = Math.max(...confidenceValues);
    const voteHistogram = Array.from({ length: 10 }, () => 0);
    samples.forEach((sample) => {
      voteHistogram[sample.predictedClass] += 1;
    });

    return {
      base,
      samples,
      stability,
      confidenceMean,
      confidenceMin,
      confidenceMax,
      voteHistogram,
      entropy: normalizedEntropy(base.probabilities),
      margin: topTwoGap(base.probabilities),
    };
  }, [hasSignal, image, model, modelKind, shiftRadius]);

  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Per-Draw Reliability</h3>
        <p className="deep-section-subtitle">
          Uses only your current drawn pixels and the active model prediction. No stale dataset confusion matrix.
        </p>
      </div>
      <div className="deep-info-card-grid">
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Equation</p>
          <p className="deep-feature-copy"><InlineMath math={'p(c|x)=\\operatorname{softmax}(z_c)'} /></p>
        </article>
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Stability metric</p>
          <p className="deep-feature-copy"><InlineMath math={'\\text{stability}=\\frac{\\#(\\hat y_{shift}=\\hat y_{base})}{N_{shifts}}'} /></p>
        </article>
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Sensitivity knob</p>
          <div className="deep-step-row">
            <button
              type="button"
              className={`deep-step-chip ${shiftRadius === 1 ? 'deep-step-chip-active' : ''}`}
              onClick={() => setShiftRadius(1)}
            >
              3x3 shifts
            </button>
            <button
              type="button"
              className={`deep-step-chip ${shiftRadius === 2 ? 'deep-step-chip-active' : ''}`}
              onClick={() => setShiftRadius(2)}
            >
              5x5 shifts
            </button>
          </div>
        </article>
      </div>

      {!analysis && (
        <p className="deep-feature-copy">
          Draw a digit and click <strong>Predict</strong> to compute reliability metrics for that exact input.
        </p>
      )}

      {analysis && (
        <>
          <div className="deep-eval-metrics">
            <article className="deep-metric-card">
              <p>Predicted digit</p>
              <strong>{analysis.base.predictedClass}</strong>
            </article>
            <article className="deep-metric-card">
              <p>Top confidence</p>
              <strong>{((analysis.base.probabilities[analysis.base.predictedClass] ?? 0) * 100).toFixed(2)}%</strong>
            </article>
            <article className="deep-metric-card">
              <p>Confidence gap (top1-top2)</p>
              <strong>{(analysis.margin * 100).toFixed(2)}%</strong>
            </article>
            <article className="deep-metric-card">
              <p>Shift stability</p>
              <strong>{(analysis.stability * 100).toFixed(1)}%</strong>
            </article>
          </div>

          <div className="deep-analysis-grid">
            <article className="deep-zone-info deep-zone-info-compact">
              <p className="deep-mini-title">Base probabilities</p>
              <div className="deep-output-bars">
                {analysis.base.probabilities.map((value, index) => (
                  <div key={`eval-prob-${index}`} className="deep-prob-row">
                    <span>{index}</span>
                    <div className="deep-prob-track"><span style={{ width: `${(value * 100).toFixed(2)}%` }} /></div>
                    <strong>{(value * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </article>
            <article className="deep-zone-info deep-zone-info-compact">
              <p className="deep-mini-title">Shift vote distribution</p>
              <div className="deep-output-bars">
                {analysis.voteHistogram.map((votes, index) => {
                  const pct = (votes / Math.max(1, analysis.samples.length)) * 100;
                  return (
                    <div key={`eval-vote-${index}`} className="deep-prob-row">
                      <span>{index}</span>
                      <div className="deep-prob-track"><span style={{ width: `${pct.toFixed(2)}%` }} /></div>
                      <strong>{votes}</strong>
                    </div>
                  );
                })}
              </div>
            </article>
          </div>

          <article className="deep-zone-info deep-zone-info-compact">
            <p className="deep-mini-title">Shift map</p>
            <div className="deep-shift-grid" style={{ gridTemplateColumns: `repeat(${(shiftRadius * 2) + 1}, minmax(0, 1fr))` }}>
              {analysis.samples.map((sample) => {
                const same = sample.predictedClass === analysis.base.predictedClass;
                const bg = same ? 'rgba(34,197,94,0.2)' : 'rgba(245,158,11,0.2)';
                return (
                  <div key={`shift-${sample.dr}-${sample.dc}`} className="deep-shift-cell" style={{ background: bg }}>
                    <span className="deep-shift-label">({sample.dr},{sample.dc})</span>
                    <strong>{sample.predictedClass}</strong>
                    <span>{(sample.confidence * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
            <p className="deep-feature-copy">
              Confidence range across shifts: <strong>{(analysis.confidenceMin * 100).toFixed(1)}%</strong> to <strong>{(analysis.confidenceMax * 100).toFixed(1)}%</strong>,
              mean <strong>{(analysis.confidenceMean * 100).toFixed(1)}%</strong>, entropy <strong>{analysis.entropy.toFixed(3)}</strong>.
            </p>
          </article>
        </>
      )}
    </section>
  );
}
