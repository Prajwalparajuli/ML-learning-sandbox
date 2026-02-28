import type { CnnPipelineStage, DeepViewMode } from '@/data/deep-learning/types';

interface CnnStageMathPanelProps {
  stage: CnnPipelineStage;
  viewMode: DeepViewMode;
  kernel: number[][];
  inputMatrix: number[][];
  convMatrix: number[][];
  reluMatrix: number[][];
  poolMatrix: number[][];
  flattened: number[];
  probabilities: { cat: number; dog: number };
  convRow: number;
  convCol: number;
}

const matrixStyle = (value: number, maxAbs: number) => {
  const intensity = Math.max(0.12, Math.min(0.95, Math.abs(value) / Math.max(maxAbs, 0.01)));
  const hue = value >= 0 ? 'hsl(var(--accent))' : 'hsl(var(--destructive))';
  return { backgroundColor: `${hue.slice(0, -1)} / ${intensity})` };
};

function Matrix({
  values,
  highlights = new Set<string>(),
  name,
  viewMode,
}: {
  values: number[][];
  highlights?: Set<string>;
  name: string;
  viewMode: DeepViewMode;
}) {
  const maxAbs = Math.max(...values.flat().map((value) => Math.abs(value)));
  return (
    <div className="deep-matrix-scroll">
      <div className="deep-matrix-grid" style={{ gridTemplateColumns: `repeat(${values[0]?.length ?? 1}, minmax(0, 28px))` }}>
        {values.map((row, r) => row.map((value, c) => (
          <div
            key={`${name}-${r}-${c}`}
            className={`deep-matrix-cell ${highlights.has(`${r}-${c}`) ? 'deep-matrix-cell-active' : ''}`}
            style={matrixStyle(value, maxAbs)}
          >
            {viewMode === 'numbers' ? value.toFixed(2) : ''}
          </div>
        )))}
      </div>
    </div>
  );
}

export function CnnStageMathPanel({
  stage,
  viewMode,
  kernel,
  inputMatrix,
  convMatrix,
  reluMatrix,
  poolMatrix,
  flattened,
  probabilities,
  convRow,
  convCol,
}: CnnStageMathPanelProps) {
  const patchHighlight = new Set<string>();
  for (let r = convRow; r < convRow + 3; r += 1) {
    for (let c = convCol; c < convCol + 3; c += 1) {
      patchHighlight.add(`${r}-${c}`);
    }
  }

  const outputHighlight = new Set<string>([`${convRow}-${convCol}`]);

  return (
    <section className="deep-panel deep-panel-primary">
      <div className="deep-section-head">
        <h4 className="deep-section-title">Stage Workspace: {stage.label}</h4>
        <p className="deep-section-subtitle">{stage.summary}</p>
      </div>

      <div className="deep-stage-visual-card">
        {stage.id === 'input' && <Matrix values={inputMatrix} name="input" viewMode={viewMode} />}

        {stage.id === 'conv' && (
          <div className="deep-matrix-stage">
            <div>
              <p className="deep-mini-title">Input patch</p>
              <Matrix values={inputMatrix} highlights={patchHighlight} name="input-patch" viewMode={viewMode} />
            </div>
            <div>
              <p className="deep-mini-title">Convolution map</p>
              <Matrix values={convMatrix} highlights={outputHighlight} name="conv" viewMode={viewMode} />
            </div>
            <div>
              <p className="deep-mini-title">Kernel</p>
              <Matrix values={kernel} name="kernel" viewMode="numbers" />
            </div>
          </div>
        )}

        {stage.id === 'relu' && <Matrix values={reluMatrix} name="relu" viewMode={viewMode} />}
        {stage.id === 'pool' && <Matrix values={poolMatrix} name="pool" viewMode={viewMode} />}

        {stage.id === 'flatten' && (
          <div className="deep-flatten-row">
            {flattened.map((value, idx) => (
              <span key={`flat-${idx}`} className="deep-flat-chip">{value.toFixed(2)}</span>
            ))}
          </div>
        )}

        {stage.id === 'dense' && (
          <div className="deep-output-bars">
            <div className="deep-prob-row">
              <span>Cat</span>
              <div className="deep-prob-track"><span style={{ width: `${(probabilities.cat * 100).toFixed(1)}%` }} /></div>
              <strong>{(probabilities.cat * 100).toFixed(1)}%</strong>
            </div>
            <div className="deep-prob-row">
              <span>Dog</span>
              <div className="deep-prob-track"><span style={{ width: `${(probabilities.dog * 100).toFixed(1)}%` }} /></div>
              <strong>{(probabilities.dog * 100).toFixed(1)}%</strong>
            </div>
          </div>
        )}
      </div>

      <div className="deep-stage-micro-math">
        <p><strong>Stage math:</strong> {stage.stage_math_summary ?? stage.summary}</p>
      </div>
      <p className="deep-feature-copy"><strong>Why this matters:</strong> {stage.summary}</p>
      <div className="deep-misconception">
        <strong>Common misconception:</strong> {stage.misconception}
      </div>
    </section>
  );
}
