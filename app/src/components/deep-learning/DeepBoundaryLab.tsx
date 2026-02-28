import { useMemo } from 'react';
import { InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';
import { useTheme } from 'next-themes';
import { runCnnInference } from '@/lib/deepLearning/cnnTrace';
import { runMlpInference } from '@/lib/deepLearning/mlpTrace';
import type { MnistCnnModel, MnistMlpModel } from '@/lib/deepLearning/modelLoader';

interface DeepBoundaryLabProps {
  modelKind: 'mnist_mlp' | 'mnist_cnn';
  model: MnistMlpModel | MnistCnnModel | null;
  image: number[][];
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

const shiftAxis = Array.from({ length: 9 }, (_, index) => index - 4);

export function DeepBoundaryLab({ modelKind, model, image }: DeepBoundaryLabProps) {
  const { resolvedTheme } = useTheme();
  const hasSignal = useMemo(() => image.some((row) => row.some((value) => value > 0.06)), [image]);

  const surface = useMemo(() => {
    if (!model || !hasSignal) return null;
    const runSnapshot = (candidate: number[][]) => (
      modelKind === 'mnist_mlp'
        ? runMlpInference(candidate, model as MnistMlpModel).snapshot
        : runCnnInference(candidate, model as MnistCnnModel).snapshot
    );

    const base = runSnapshot(image);
    const targetClass = base.predictedClass;
    const z = shiftAxis.map(() => Array.from({ length: shiftAxis.length }, () => 0));
    const changedX: number[] = [];
    const changedY: number[] = [];

    for (let r = 0; r < shiftAxis.length; r += 1) {
      for (let c = 0; c < shiftAxis.length; c += 1) {
        const shifted = shiftImage(image, shiftAxis[r], shiftAxis[c]);
        const snapshot = runSnapshot(shifted);
        z[r][c] = snapshot.probabilities[targetClass] ?? 0;
        if (snapshot.predictedClass !== targetClass) {
          changedX.push(shiftAxis[c]);
          changedY.push(shiftAxis[r]);
        }
      }
    }

    return { base, targetClass, z, changedX, changedY };
  }, [hasSignal, image, model, modelKind]);

  const isDark = resolvedTheme === 'dark';

  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Local Decision Surface</h3>
        <p className="deep-section-subtitle">Confidence landscape around your current drawing under translation perturbations.</p>
      </div>
      <div className="deep-info-card-grid">
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">What This Shows</p>
          <p className="deep-feature-copy">How stable your current prediction is when the same digit is shifted a few pixels.</p>
        </article>
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Equation Focus</p>
          <p className="deep-feature-copy"><InlineMath math={'p(\\hat{c}\\mid x_{\\Delta r,\\Delta c})'} /></p>
        </article>
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Interpretation</p>
          <p className="deep-feature-copy">Lighter regions indicate the predicted class stays confident under small input shifts.</p>
        </article>
      </div>
      {!surface && (
        <p className="deep-feature-copy">
          Draw a digit and click <strong>Predict</strong> to build a local decision surface for that exact input.
        </p>
      )}
      {surface && (
        <>
          <p className="deep-feature-copy">
            Active model: <strong>{modelKind === 'mnist_mlp' ? 'MNIST MLP' : 'MNIST CNN'}</strong> |
            base class <strong>{surface.targetClass}</strong> with confidence{' '}
            <strong>{((surface.base.probabilities[surface.targetClass] ?? 0) * 100).toFixed(2)}%</strong>.
          </p>
          <div className="plot-wrap code-block h-[320px] overflow-hidden mt-2">
            <Plot
              data={[
                {
                  x: shiftAxis,
                  y: shiftAxis,
                  z: surface.z,
                  type: 'contour',
                  contours: { coloring: 'fill', showlines: false, start: 0, end: 1, size: 0.1 },
                  colorscale: [
                    [0, isDark ? 'rgba(59,130,246,0.24)' : 'rgba(59,130,246,0.2)'],
                    [0.5, isDark ? 'rgba(248,250,252,0.08)' : 'rgba(248,250,252,0.38)'],
                    [1, isDark ? 'rgba(244,63,94,0.24)' : 'rgba(244,63,94,0.2)'],
                  ],
                  showscale: false,
                  hovertemplate: 'dc: %{x}<br>dr: %{y}<br>p: %{z:.3f}<extra></extra>',
                },
                {
                  x: shiftAxis,
                  y: shiftAxis,
                  z: surface.z,
                  type: 'contour',
                  contours: { start: 0.5, end: 0.5, size: 1, coloring: 'lines', showlabels: false },
                  line: { color: isDark ? '#e2e8f0' : '#0f172a', width: 1.8 },
                  showscale: false,
                  hoverinfo: 'skip',
                },
                {
                  x: [0],
                  y: [0],
                  mode: 'markers',
                  type: 'scatter',
                  marker: {
                    color: isDark ? '#f8fafc' : '#0f172a',
                    symbol: 'star',
                    size: 12,
                    line: { color: isDark ? '#22d3ee' : '#0ea5e9', width: 1.6 },
                  },
                  name: 'Current draw',
                },
                {
                  x: surface.changedX,
                  y: surface.changedY,
                  mode: 'markers',
                  type: 'scatter',
                  marker: {
                    color: isDark ? '#fbbf24' : '#b45309',
                    size: 6,
                    symbol: 'x',
                  },
                  name: 'Class switch',
                },
              ]}
              layout={{
                margin: { l: 46, r: 10, t: 10, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: { text: 'Horizontal shift (px)' }, range: [-4.2, 4.2], dtick: 1 },
                yaxis: { title: { text: 'Vertical shift (px)' }, range: [-4.2, 4.2], dtick: 1 },
                legend: { x: 0.02, y: 0.98, bgcolor: isDark ? 'rgba(15,23,42,0.62)' : 'rgba(255,255,255,0.72)' },
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </>
      )}
    </section>
  );
}
