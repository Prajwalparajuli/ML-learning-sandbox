import { useEffect, useMemo, useState } from 'react';
import { InlineMath } from 'react-katex';
import { convValid } from '@/lib/deepLearning/mnistPreprocess';

interface KernelLabExplorerProps {
  image: number[][];
}

const defaultKernel = [
  [1, 0, -1],
  [1, 0, -1],
  [1, 0, -1],
];

const flattenKernel = (kernel: number[][]): number[] => kernel.flat();

const formatTerm = (pixel: number, weight: number): string => `${pixel.toFixed(2)}×${weight.toFixed(2)}`;

export function KernelLabExplorer({ image }: KernelLabExplorerProps) {
  const [kernel, setKernel] = useState<number[][]>(defaultKernel);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(220);

  const convMap = useMemo(() => convValid(image, kernel), [image, kernel]);
  const rows = convMap.length;
  const cols = convMap[0]?.length ?? 1;
  const totalSteps = Math.max(1, rows * cols);
  const clampedStep = Math.max(0, Math.min(step, totalSteps - 1));
  const currentRow = Math.floor(clampedStep / cols);
  const currentCol = clampedStep % cols;

  const patch = useMemo(() => {
    return Array.from({ length: 3 }, (_, r) =>
      Array.from({ length: 3 }, (_, c) => image[currentRow + r]?.[currentCol + c] ?? 0)
    );
  }, [currentCol, currentRow, image]);

  const equationTerms = useMemo(() => {
    const terms: string[] = [];
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 3; c += 1) {
        terms.push(formatTerm(patch[r][c], kernel[r][c]));
      }
    }
    return terms;
  }, [kernel, patch]);

  const currentValue = convMap[currentRow]?.[currentCol] ?? 0;

  const progressiveMap = useMemo(() => {
    return convMap.map((row, r) => row.map((value, c) => {
      const idx = r * cols + c;
      return idx <= clampedStep ? value : 0;
    }));
  }, [clampedStep, cols, convMap]);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setStep((prev) => (prev >= totalSteps - 1 ? 0 : prev + 1));
    }, speedMs);
    return () => window.clearInterval(timer);
  }, [playing, speedMs, totalSteps]);

  return (
    <section className="deep-panel deep-panel-primary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Kernel Lab (Educational Simulation)</h3>
        <p className="deep-section-subtitle">Editable 3x3 kernel with stepwise sliding-window convolution animation.</p>
      </div>
      <p className="deep-feature-copy"><strong>Badge:</strong> Educational simulation (not model-derived prediction path).</p>

      <div className="deep-step-row">
        <button type="button" className="deep-step-chip" onClick={() => setStep((prev) => Math.max(0, prev - 1))}>Prev</button>
        <button type="button" className="deep-step-chip" onClick={() => setStep((prev) => (prev >= totalSteps - 1 ? 0 : prev + 1))}>Step</button>
        <button
          type="button"
          className={`deep-step-chip ${playing ? 'deep-step-chip-active' : ''}`}
          onClick={() => setPlaying((prev) => !prev)}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <label className="deep-control">
          <span>Speed</span>
          <select value={speedMs} onChange={(event) => setSpeedMs(Number(event.target.value))}>
            <option value={140}>Fast</option>
            <option value={220}>Normal</option>
            <option value={360}>Slow</option>
          </select>
        </label>
        <button type="button" className="deep-step-chip" onClick={() => setStep(0)}>Reset</button>
      </div>
      <div className="deep-info-card-grid">
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">What You Are Seeing</p>
          <p className="deep-feature-copy">A single 3x3 kernel sliding across your image and writing one output value per location.</p>
        </article>
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Equation Focus</p>
          <p className="deep-feature-copy"><InlineMath math={'\\sum_{i,j}(x_{ij}k_{ij}) = y_{r,c}'} /></p>
        </article>
        <article className="deep-zone-info deep-zone-info-compact">
          <p className="deep-mini-title">Why This Lab Exists</p>
          <p className="deep-feature-copy">Kernel editing is educational only and is intentionally separated from real model inference.</p>
        </article>
      </div>

      <div className="deep-zone-grid">
        <div className="deep-zone-visual">
          <article className="deep-matrix-card">
            <p className="deep-mini-title">Kernel Editor (3x3)</p>
            <div className="deep-kernel-editor">
              {kernel.map((row, r) => row.map((value, c) => (
                <input
                  key={`k-${r}-${c}`}
                  type="number"
                  step={0.1}
                  min={-2.5}
                  max={2.5}
                  value={value}
                  onChange={(event) => {
                    const next = kernel.map((line) => line.slice());
                    next[r][c] = Number(event.target.value);
                    setKernel(next);
                    setStep(0);
                  }}
                />
              )))}
            </div>
            <p className="deep-feature-copy">Kernel vector: [{flattenKernel(kernel).map((v) => v.toFixed(2)).join(', ')}]</p>
          </article>

          <article className="deep-matrix-card">
            <p className="deep-mini-title">Output Feature Map (Progressive)</p>
            <div className="deep-matrix-scroll">
              <div className="deep-matrix-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 12px))` }}>
                {progressiveMap.map((row, r) => row.map((value, c) => {
                  const active = r === currentRow && c === currentCol;
                  const alpha = Math.max(0.1, Math.min(0.9, Math.abs(value) / 2.8));
                  return (
                    <div
                      key={`out-${r}-${c}`}
                      className={`deep-matrix-cell deep-matrix-cell-compact ${active ? 'deep-matrix-cell-active' : ''}`}
                      style={{
                        background: value >= 0
                          ? `rgba(244, 63, 94, ${alpha.toFixed(3)})`
                          : `rgba(59, 130, 246, ${alpha.toFixed(3)})`,
                      }}
                    />
                  );
                }))}
              </div>
            </div>
          </article>
        </div>

        <div className="deep-zone-info">
          <p className="deep-mini-title">Live Equation</p>
          <p className="deep-feature-copy">Window at row {currentRow}, col {currentCol}</p>
          <p className="deep-feature-copy">{equationTerms.join(' + ')}</p>
          <p className="deep-feature-copy">Result: <strong>{currentValue.toFixed(4)}</strong></p>
        </div>
      </div>
    </section>
  );
}
