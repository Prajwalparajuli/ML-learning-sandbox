import type { DeepStepDelta, MlpStepMath } from '@/data/deep-learning/types';

interface MlpMathPanelProps {
  activation: 'relu' | 'sigmoid';
  stepLabel: string;
  math: MlpStepMath;
  stepDeltas: DeepStepDelta[];
}

function formatVector(values: number[]) {
  return `[${values.map((value) => value.toFixed(2)).join(', ')}]`;
}

export function MlpMathPanel({ activation, stepLabel, math, stepDeltas }: MlpMathPanelProps) {
  const normalizedX = Math.max(-4, Math.min(4, math.activationPoint.z));
  const cx = ((normalizedX + 4) / 8) * 160 + 10;
  const cy = activation === 'relu'
    ? 90 - Math.max(0, Math.min(1, math.activationPoint.a / 4)) * 72
    : 90 - Math.max(0, Math.min(1, math.activationPoint.a)) * 72;
  const deltaText = stepDeltas
    .map((item) => {
      const dir = item.delta >= 0 ? 'increased' : 'decreased';
      const magnitude = Math.abs(item.delta).toFixed(4);
      return `${item.metric} ${dir} by ${magnitude}`;
    })
    .join(', ');

  return (
    <section className="deep-panel deep-panel-primary">
      <div className="deep-section-head">
        <h4 className="deep-section-title">Math Lens: {stepLabel}</h4>
        <p className="deep-section-subtitle">Compact equation view for the active layer transition.</p>
      </div>
      <div className="deep-math-grid">
        <div>
          <p className="deep-mini-title">Input vector</p>
          <code className="deep-math-code">{formatVector(math.inputVector)}</code>
        </div>
        <div>
          <p className="deep-mini-title">Weight vector</p>
          <code className="deep-math-code">{formatVector(math.weightVector)}</code>
        </div>
        <div>
          <p className="deep-mini-title">Bias</p>
          <code className="deep-math-code">{math.bias.toFixed(3)}</code>
        </div>
      </div>
      <p className="deep-feature-copy">
        <strong>z = W·x + b</strong> = {math.z.toFixed(4)} | <strong>{activation}(z)</strong> = {math.activationOutput.toFixed(4)}
      </p>
      <p className="deep-feature-copy">Compared to previous step: {deltaText}.</p>
      <svg viewBox="0 0 180 100" className="deep-activation-plot deep-activation-plot-compact" role="img" aria-label={`${activation} activation curve`}>
        <line x1="10" y1="90" x2="170" y2="90" className="deep-plot-axis" />
        <line x1="10" y1="10" x2="10" y2="90" className="deep-plot-axis" />
        {activation === 'relu' ? (
          <>
            <line x1="10" y1="90" x2="90" y2="90" className="deep-plot-curve" />
            <line x1="90" y1="90" x2="170" y2="18" className="deep-plot-curve" />
          </>
        ) : (
          <path d="M10,88 C55,88 54,12 90,12 C126,12 125,88 170,88" className="deep-plot-curve" fill="none" />
        )}
        <circle cx={cx} cy={cy} r="4" className="deep-plot-point" />
      </svg>
    </section>
  );
}
