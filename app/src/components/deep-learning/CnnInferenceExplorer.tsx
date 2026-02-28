import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { InlineMath } from 'react-katex';
import type { DlInferenceMode } from '@/data/deep-learning/types';
import { runCnnInference } from '@/lib/deepLearning/cnnTrace';
import type { MnistCnnModel } from '@/lib/deepLearning/modelLoader';

interface CnnInferenceExplorerProps {
  image: number[][];
  model: MnistCnnModel | null;
  topChannels: number;
  predictNonce: number;
  inferenceMode: DlInferenceMode;
  experienceMode?: 'predict' | 'learn';
  compactPredict?: boolean;
}

const stageLabels = ['Input', 'Conv Window', 'ReLU', 'Pool', 'Flatten', 'Dense'];
const stageCaptions = [
  'Input: your 28x28 drawing is normalized and fed to CNN filters.',
  'Conv Window: slide the 3x3 filter step-by-step and fill the output map progressively.',
  'ReLU: negative responses are set to zero.',
  'Pool: 2x2 max pooling retains strongest local activations.',
  'Flatten: pooled maps are unrolled into one feature vector.',
  'Dense: logits are converted into probabilities.',
];

const stageLearningNotes = [
  'What: raw pixels enter convolution filters. Why: early filters detect local strokes.',
  'What: one kernel moves across the image. Why: weight sharing finds the same pattern anywhere.',
  'What: ReLU zeros negatives. Why: weak negative evidence is suppressed.',
  'What: max-pool downsamples activations. Why: strongest local evidence is retained.',
  'What: maps flatten into one vector. Why: dense layer expects a single feature vector.',
  'What: logits map to probabilities. Why: confidence bars summarize the final class decision.',
];

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const convValidStride = (
  input: number[][],
  kernel: number[][],
  stride: number
): number[][] => {
  const rows = input.length;
  const cols = input[0]?.length ?? 0;
  const kernelRows = kernel.length;
  const kernelCols = kernel[0]?.length ?? 0;
  if (rows < kernelRows || cols < kernelCols) return [[0]];
  const outRows = Math.floor((rows - kernelRows) / stride) + 1;
  const outCols = Math.floor((cols - kernelCols) / stride) + 1;
  const out = Array.from({ length: outRows }, () => Array.from({ length: outCols }, () => 0));
  for (let r = 0; r < outRows; r += 1) {
    for (let c = 0; c < outCols; c += 1) {
      let sum = 0;
      const rr = r * stride;
      const cc = c * stride;
      for (let kr = 0; kr < kernelRows; kr += 1) {
        for (let kc = 0; kc < kernelCols; kc += 1) {
          sum += (input[rr + kr]?.[cc + kc] ?? 0) * (kernel[kr]?.[kc] ?? 0);
        }
      }
      out[r][c] = sum;
    }
  }
  return out;
};

const reluFloor = (input: number[][], floor: number): number[][] =>
  input.map((row) => row.map((value) => (value > floor ? value : 0)));

const pool2x2 = (input: number[][], mode: 'max' | 'avg'): number[][] => {
  const rows = input.length;
  const cols = input[0]?.length ?? 0;
  if (rows < 2 || cols < 2) return [[0]];
  const outRows = Math.floor(rows / 2);
  const outCols = Math.floor(cols / 2);
  const out = Array.from({ length: outRows }, () => Array.from({ length: outCols }, () => 0));
  for (let r = 0; r < outRows; r += 1) {
    for (let c = 0; c < outCols; c += 1) {
      const rr = r * 2;
      const cc = c * 2;
      const v0 = input[rr][cc];
      const v1 = input[rr][cc + 1];
      const v2 = input[rr + 1][cc];
      const v3 = input[rr + 1][cc + 1];
      out[r][c] = mode === 'avg' ? (v0 + v1 + v2 + v3) / 4 : Math.max(v0, v1, v2, v3);
    }
  }
  return out;
};

const toCellStyle = (value: number, maxAbs: number): CSSProperties => {
  const alpha = Math.max(0.1, Math.min(0.95, Math.abs(value) / Math.max(0.0001, maxAbs)));
  if (value >= 0) return { background: `rgba(239, 68, 68, ${alpha.toFixed(3)})` };
  return { background: `rgba(59, 130, 246, ${alpha.toFixed(3)})` };
};

function MatrixCard({
  title,
  matrix,
  highlight,
}: {
  title: string;
  matrix: number[][];
  highlight?: Set<string>;
}) {
  const maxAbs = Math.max(1e-5, ...matrix.flat().map((value) => Math.abs(value)));
  const columns = matrix[0]?.length ?? 1;
  const cellSize = columns >= 24 ? 6 : columns >= 16 ? 8 : columns >= 8 ? 10 : 14;
  const cellGap = columns >= 24 ? 1 : columns >= 16 ? 1 : columns >= 8 ? 2 : 3;
  return (
    <article className="deep-matrix-card">
      <p className="deep-mini-title">{title}</p>
      <div className="deep-matrix-scroll">
        <div
          className="deep-matrix-grid"
          style={{
            gridTemplateColumns: `repeat(${columns}, minmax(0, ${cellSize}px))`,
            gap: `${cellGap}px`,
          }}
        >
          {matrix.map((row, r) => row.map((value, c) => (
            <div
              key={`${title}-${r}-${c}`}
              className={`deep-matrix-cell deep-matrix-cell-compact ${highlight?.has(`${r}-${c}`) ? 'deep-matrix-cell-active' : ''}`}
              style={{
                ...toCellStyle(value, maxAbs),
                width: `${cellSize}px`,
                height: `${cellSize}px`,
                minHeight: `${cellSize}px`,
              }}
            />
          )))}
        </div>
      </div>
    </article>
  );
}

export function CnnInferenceExplorer({
  image,
  model,
  topChannels,
  predictNonce,
  inferenceMode,
  experienceMode = 'learn',
  compactPredict = false,
}: CnnInferenceExplorerProps) {
  const isLearnMode = experienceMode === 'learn';
  const isPredictCompact = !isLearnMode && compactPredict;
  const lastStageIndex = stageLabels.length - 1;
  const [stage, setStage] = useState(1);
  const [scanStep, setScanStep] = useState(0);
  const [windowPlaying, setWindowPlaying] = useState(false);
  const [windowSpeedMs, setWindowSpeedMs] = useState(75);
  const [layerPlaying, setLayerPlaying] = useState(false);
  const [kernelScale, setKernelScale] = useState(1);
  const [stride, setStride] = useState(1);
  const [reluThreshold, setReluThreshold] = useState(0);
  const [poolMode, setPoolMode] = useState<'max' | 'avg'>('max');
  const [hasInteracted, setHasInteracted] = useState(false);
  const hasSignal = useMemo(() => image.some((row) => row.some((value) => value > 0.06)), [image]);

  const inference = useMemo(() => {
    if (!model || !hasSignal) return null;
    return runCnnInference(image, model, { topChannels, inferenceMode });
  }, [hasSignal, image, inferenceMode, model, topChannels]);

  const selectedChannel = inference?.trace.conv[0]?.channel ?? 0;
  const selectedKernel = useMemo(() => {
    return model?.kernels[selectedChannel] ?? [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
  }, [model, selectedChannel]);

  const scaledKernel = useMemo(
    () => selectedKernel.map((row) => row.map((value) => value * kernelScale)),
    [kernelScale, selectedKernel]
  );
  const convMatrix = useMemo(() => {
    if (!inference) return [[]];
    if (!isLearnMode) return inference.trace.conv[0]?.values ?? [[]];
    return convValidStride(image, scaledKernel, stride);
  }, [image, inference, isLearnMode, scaledKernel, stride]);
  const reluStageMatrix = useMemo(() => {
    if (!inference) return [[]];
    if (!isLearnMode) return inference.trace.relu[0]?.values ?? [[]];
    return reluFloor(convMatrix, reluThreshold);
  }, [convMatrix, inference, isLearnMode, reluThreshold]);
  const poolStageMatrix = useMemo(() => {
    if (!inference) return [[]];
    if (!isLearnMode) return inference.trace.pool[0]?.values ?? [[]];
    return pool2x2(reluStageMatrix, poolMode);
  }, [inference, isLearnMode, poolMode, reluStageMatrix]);
  const convRows = convMatrix.length || 1;
  const convCols = convMatrix[0]?.length || 1;
  const maxScan = convRows * convCols;
  const activeScan = Math.min(scanStep, Math.max(0, maxScan - 1));
  const scanRow = Math.floor(activeScan / convCols);
  const scanCol = activeScan % convCols;
  const patchRow = scanRow * stride;
  const patchCol = scanCol * stride;

  const patch = useMemo(() => {
    return Array.from({ length: 3 }, (_, r) =>
      Array.from({ length: 3 }, (_, c) => image[patchRow + r]?.[patchCol + c] ?? 0)
    );
  }, [image, patchCol, patchRow]);

  const patchHighlight = useMemo(() => {
    const set = new Set<string>();
    for (let r = patchRow; r < patchRow + 3; r += 1) {
      for (let c = patchCol; c < patchCol + 3; c += 1) set.add(`${r}-${c}`);
    }
    return set;
  }, [patchCol, patchRow]);

  const progressiveConv = useMemo(() => {
    if (!convMatrix[0]) return [];
    return convMatrix.map((row, r) => row.map((value, c) => {
      const idx = r * convCols + c;
      return idx <= activeScan ? value : 0;
    }));
  }, [activeScan, convCols, convMatrix]);

  const matrixTerms = useMemo(() => {
    return Array.from({ length: 3 }, (_, r) =>
      Array.from({ length: 3 }, (_, c) => {
        const pixel = patch[r][c] ?? 0;
        const kernel = scaledKernel[r][c] ?? 0;
        return {
          row: r,
          col: c,
          pixel,
          kernel,
          product: pixel * kernel,
        };
      })
    );
  }, [patch, scaledKernel]);
  const activeTermIndex = activeScan % 9;
  const flattenVector = useMemo(
    () => (isLearnMode ? poolStageMatrix.flat() : (inference?.flattened ?? [])),
    [inference, isLearnMode, poolStageMatrix]
  );
  const inputStats = useMemo(() => {
    const rows = image.length;
    const cols = image[0]?.length ?? 0;
    let activePixels = 0;
    let sum = 0;
    let max = 0;
    let rowWeighted = 0;
    let colWeighted = 0;
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const value = image[r]?.[c] ?? 0;
        sum += value;
        if (value > 0.06) activePixels += 1;
        if (value > max) max = value;
        rowWeighted += r * value;
        colWeighted += c * value;
      }
    }
    const density = rows * cols > 0 ? activePixels / (rows * cols) : 0;
    const centroidRow = sum > 1e-6 ? rowWeighted / sum : 13.5;
    const centroidCol = sum > 1e-6 ? colWeighted / sum : 13.5;
    return {
      activePixels,
      totalPixels: rows * cols,
      density,
      max,
      centroidRow,
      centroidCol,
    };
  }, [image]);
  const reluPositive = useMemo(
    () => reluStageMatrix.flat().filter((value) => value > 0).length,
    [reluStageMatrix]
  );
  const compactLayerTiles = useMemo(() => {
    const convValue = convMatrix[scanRow]?.[scanCol] ?? 0;
    const poolRows = poolStageMatrix.length;
    const poolCols = poolStageMatrix[0]?.length ?? 0;
    const topProb = inference ? Math.max(...inference.snapshot.probabilities) : 0;
    return [
      {
        title: 'Input Detail',
        body: `${inputStats.activePixels}/${inputStats.totalPixels} active pixels (${(inputStats.density * 100).toFixed(1)}%). Centroid (${inputStats.centroidCol.toFixed(1)}, ${inputStats.centroidRow.toFixed(1)}).`,
      },
      {
        title: 'Conv Window',
        body: `Patch (${patchRow}, ${patchCol}) with 3x3 kernel gives ${convValue.toFixed(4)} at output (${scanRow}, ${scanCol}).`,
      },
      {
        title: 'ReLU',
        body: `${reluPositive} positive activations kept. Negatives are clipped to 0.`,
      },
      {
        title: 'MaxPool',
        body: `Downsampled to ${poolRows}x${poolCols} by retaining strongest 2x2 responses.`,
      },
      {
        title: 'Flatten',
        body: `${flattenVector.length} features unrolled into a dense-ready vector.`,
      },
      {
        title: 'Dense + Softmax',
        body: inference
          ? `Class ${inference.snapshot.predictedClass} selected with ${(topProb * 100).toFixed(1)}% confidence.`
          : 'Dense logits are converted into normalized class probabilities.',
      },
    ];
  }, [
    convMatrix,
    flattenVector.length,
    inference,
    inputStats.activePixels,
    inputStats.centroidCol,
    inputStats.centroidRow,
    inputStats.density,
    inputStats.totalPixels,
    patchCol,
    patchRow,
    poolStageMatrix,
    reluPositive,
    scanCol,
    scanRow,
  ]);

  const resetAll = () => {
    setHasInteracted(true);
    setLayerPlaying(false);
    setWindowPlaying(false);
    setStage(0);
    setScanStep(0);
  };

  const stepLayer = () => {
    setHasInteracted(true);
    setLayerPlaying(false);
    setWindowPlaying(false);
    setStage((prev) => Math.min(lastStageIndex, prev + 1));
  };

  const stepWindow = () => {
    setHasInteracted(true);
    setScanStep((prev) => Math.min(maxScan - 1, prev + 1));
  };

  const toggleWindowPlay = () => {
    setHasInteracted(true);
    setLayerPlaying(false);
    setWindowPlaying((prev) => !prev);
  };

  const toggleLayerPlay = () => {
    setHasInteracted(true);
    setWindowPlaying(false);
    setLayerPlaying((previous) => !previous);
  };

  useEffect(() => {
    if (!windowPlaying || stage !== 1 || !inference) return undefined;
    const timer = window.setInterval(() => {
      setScanStep((prev) => {
        if (prev >= maxScan - 1) {
          setWindowPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, windowSpeedMs);
    return () => window.clearInterval(timer);
  }, [inference, maxScan, stage, windowPlaying, windowSpeedMs]);

  useEffect(() => {
    if (!layerPlaying || !inference) return undefined;
    const timer = window.setInterval(() => {
      setHasInteracted(true);
      if (stage === 1) {
        setScanStep((previous) => {
          if (previous < maxScan - 1) return previous + 1;
          setStage(2);
          return previous;
        });
        return;
      }
      setStage((previous) => {
        if (previous >= lastStageIndex) {
          setLayerPlaying(false);
          return previous;
        }
        return previous + 1;
      });
    }, 420);
    return () => window.clearInterval(timer);
  }, [inference, lastStageIndex, layerPlaying, maxScan, stage]);

  useEffect(() => {
    setStage(1);
    setScanStep(0);
    setWindowPlaying(false);
    setLayerPlaying(false);
    setHasInteracted(false);
    setKernelScale(1);
    setStride(1);
    setReluThreshold(0);
    setPoolMode('max');
  }, [predictNonce]);

  return (
    <section className={`deep-panel deep-panel-primary ${isPredictCompact ? 'studio-cnn-predict-panel' : ''}`}>
      <div className={`deep-section-head ${isPredictCompact ? 'studio-cnn-predict-head' : ''}`}>
        <h3 className="deep-section-title">
          {isLearnMode ? 'CNN Learn Flow' : 'CNN Prediction Studio'}
        </h3>
        {isLearnMode ? <p className="deep-section-subtitle">Stage-by-stage CNN flow.</p> : null}
      </div>

      {isPredictCompact ? (
        <div className="studio-cnn-compact-flow">
          {stageLabels.map((label, index) => (
            <span key={`cnn-stage-pill-${label}`} className={`studio-cnn-compact-pill ${stage === index ? 'is-active' : ''}`}>
              {label}
            </span>
          ))}
        </div>
      ) : (
        <div className={`studio-cnn-stage-strip ${isPredictCompact ? 'is-compact' : ''}`}>
          {stageLabels.map((label, index) => (
            <button
              key={`cnn-stage-strip-${label}`}
              type="button"
              className={`studio-cnn-stage-chip ${stage === index ? 'is-active' : ''}`}
              onClick={() => {
                setHasInteracted(true);
                setStage(index);
                if (index !== 1) setWindowPlaying(false);
              }}
              disabled={!inference}
            >
              <span>{index + 1}</span>
              {label}
            </button>
          ))}
        </div>
      )}

      <div className={`deep-step-row ${isPredictCompact ? 'studio-cnn-step-row-compact' : ''}`}>
        <button
          type="button"
          className="deep-step-chip"
          onClick={resetAll}
          disabled={!inference}
        >
          {'|<'}
        </button>
        <button
          type="button"
          className="deep-step-chip"
          onClick={isPredictCompact ? (stage === 1 ? stepWindow : stepLayer) : stepLayer}
          disabled={!inference || (stage !== 1 && stage >= lastStageIndex) || (stage === 1 && scanStep >= maxScan - 1)}
        >
          Step
        </button>
        {(isLearnMode || isPredictCompact) && (
          <button
            type="button"
            className={`deep-step-chip ${(stage === 1 ? windowPlaying : layerPlaying) ? 'deep-step-chip-active' : ''}`}
            onClick={() => {
              if (stage === 1) {
                toggleWindowPlay();
                return;
              }
              toggleLayerPlay();
            }}
            disabled={!inference}
          >
            {(stage === 1 ? windowPlaying : layerPlaying)
              ? (stage === 1 ? 'Pause Window' : 'Pause')
              : (stage === 1 ? 'Play Window' : 'Play')}
          </button>
        )}
        {!isPredictCompact && <span className="deep-pill">Step <strong>{stage + 1}</strong>/{stageLabels.length}</span>}
        {stage >= lastStageIndex && inference && !isPredictCompact && <span className="deep-pill"><strong>Done</strong></span>}
        {stage === 1 && !isPredictCompact && (
          <>
            <button
              type="button"
              className="deep-step-chip"
              onClick={stepWindow}
              disabled={!inference || scanStep >= maxScan - 1}
            >
              Step Window
            </button>
            <button
              type="button"
              className={`deep-step-chip ${windowPlaying ? 'deep-step-chip-active' : ''}`}
              onClick={toggleWindowPlay}
              disabled={!inference || scanStep >= maxScan - 1}
            >
              {windowPlaying ? 'Pause Window' : 'Play Window'}
            </button>
            <button
              type="button"
              className="deep-step-chip"
              onClick={() => {
                setHasInteracted(true);
                setScanStep(0);
              }}
              disabled={!inference}
            >
              Reset Window
            </button>
            <label className="deep-control">
              <span>Speed</span>
              <select
                value={windowSpeedMs}
                onChange={(event) => {
                  setHasInteracted(true);
                  setLayerPlaying(false);
                  setWindowSpeedMs(Number(event.target.value));
                }}
              >
                <option value={45}>Fast</option>
                <option value={75}>Normal</option>
                <option value={140}>Slow</option>
              </select>
            </label>
            <span className="deep-pill">Window <strong>{Math.min(maxScan, activeScan + 1)}</strong>/{maxScan}</span>
          </>
        )}
        {isPredictCompact && stage === 1 && (
          <>
            <label className="deep-control studio-cnn-speed-control-compact">
              <span>Speed</span>
              <select
                value={windowSpeedMs}
                onChange={(event) => {
                  setHasInteracted(true);
                  setLayerPlaying(false);
                  setWindowSpeedMs(Number(event.target.value));
                }}
              >
                <option value={45}>Fast</option>
                <option value={75}>Normal</option>
                <option value={140}>Slow</option>
              </select>
            </label>
            <span className="deep-pill studio-cnn-pill-compact">Window <strong>{Math.min(maxScan, activeScan + 1)}</strong>/{maxScan}</span>
          </>
        )}
        {isPredictCompact && stage !== 1 && (
          <span className="deep-pill studio-cnn-pill-compact">Stage <strong>{stage + 1}</strong>/{stageLabels.length}</span>
        )}
      </div>
      <p className={`deep-feature-copy ${isPredictCompact ? 'studio-cnn-stage-caption-compact' : ''}`}>
        {stageCaptions[stage]}
      </p>

      {isLearnMode && (
        <div className="studio-cnn-hyper-grid">
          <label className="deep-control">
            <span>Kernel scale</span>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.05"
              value={kernelScale}
              onChange={(event) => {
                setHasInteracted(true);
                setKernelScale(clamp(Number(event.target.value), 0.5, 2));
              }}
            />
          </label>
          <label className="deep-control">
            <span>Stride</span>
            <select
              value={stride}
              onChange={(event) => {
                setHasInteracted(true);
                setStride(clamp(Number(event.target.value), 1, 2));
                setScanStep(0);
              }}
            >
              <option value={1}>1 (dense scan)</option>
              <option value={2}>2 (faster scan)</option>
            </select>
          </label>
          <label className="deep-control">
            <span>ReLU threshold</span>
            <input
              type="range"
              min="-0.5"
              max="0.5"
              step="0.02"
              value={reluThreshold}
              onChange={(event) => {
                setHasInteracted(true);
                setReluThreshold(clamp(Number(event.target.value), -0.5, 0.5));
              }}
            />
          </label>
          <label className="deep-control">
            <span>Pool mode</span>
            <select
              value={poolMode}
              onChange={(event) => {
                setHasInteracted(true);
                setPoolMode(event.target.value as 'max' | 'avg');
              }}
            >
              <option value="max">Max Pool</option>
              <option value="avg">Average Pool</option>
            </select>
          </label>
        </div>
      )}

      {isLearnMode ? (
        <div className="deep-info-card-grid">
          <article className="deep-zone-info deep-zone-info-compact">
            <p className="deep-mini-title">Animation Talk</p>
            <p className="deep-feature-copy">{stageLearningNotes[stage]}</p>
          </article>
          <article className="deep-zone-info deep-zone-info-compact">
            <p className="deep-mini-title">Equation Focus</p>
            <p className="deep-feature-copy">
              <InlineMath
                math={
                  stage === 1
                    ? `\\sum (x_{ij}k_{ij}) = ${convMatrix[scanRow]?.[scanCol]?.toFixed(4) ?? '0.0000'}`
                    : stage === 2
                      ? '\\operatorname{ReLU}(z)=\\max(0,z)'
                      : stage === 3
                        ? '\\operatorname{pool}=\\max_{2\\times2}(\\cdot)'
                        : stage === 5
                          ? 'p=\\operatorname{softmax}(\\ell)'
                          : 'x \\to \\operatorname{conv} \\to \\operatorname{ReLU} \\to \\operatorname{pool} \\to \\operatorname{dense}'
                }
              />
            </p>
            {isLearnMode && (
              <p className="deep-feature-copy">
                Kernel scale: <strong>{kernelScale.toFixed(2)}</strong> | Stride: <strong>{stride}</strong> | ReLU floor:{' '}
                <strong>{reluThreshold.toFixed(2)}</strong> | Pool: <strong>{poolMode}</strong>
              </p>
            )}
          </article>
        </div>
      ) : null}

      {!inference && (
        <p className="deep-feature-copy">
          Draw a digit, then click <strong>Predict</strong>.
        </p>
      )}

      {inference && !isPredictCompact && (
        <div className={`deep-zone-grid ${isPredictCompact ? 'studio-cnn-zone-grid-compact' : ''}`}>
          <div className="deep-zone-visual deep-zone-visual-large">
            {stage === 0 && <MatrixCard title="Input (28x28)" matrix={image} />}
            {stage === 1 && (
              <div className="deep-cnn-window-grid">
                <MatrixCard title="Input + active patch" matrix={image} highlight={patchHighlight} />
                <MatrixCard title="Kernel (3x3)" matrix={scaledKernel} />
                <div className="studio-cnn-output-cluster">
                  <MatrixCard title="Output feature map (progressive)" matrix={progressiveConv} />
                  <article className="studio-cnn-math-tooltip">
                    <p className="deep-mini-title">Live 3x3 Math</p>
                    <div className="studio-cnn-math-matrix">
                      {matrixTerms.map((row, rowIndex) => (
                        row.map((cell, colIndex) => {
                          const flatIndex = rowIndex * 3 + colIndex;
                          return (
                            <div
                              key={`cnn-math-${rowIndex}-${colIndex}`}
                              className={`studio-cnn-math-cell ${flatIndex === activeTermIndex ? 'is-active' : ''}`}
                            >
                              <span>{cell.pixel.toFixed(2)} x {cell.kernel.toFixed(2)}</span>
                              <strong>{cell.product.toFixed(3)}</strong>
                            </div>
                          );
                        })
                      ))}
                    </div>
                    <p className="studio-cnn-math-sum">
                      Sum = <strong>{convMatrix[scanRow]?.[scanCol]?.toFixed(4) ?? '0.0000'}</strong>
                    </p>
                    <div className="studio-cnn-op-list">
                      <p><strong>Conv:</strong> <InlineMath math={'\\sum (x_{ij}k_{ij})'} /></p>
                      <p><strong>ReLU:</strong> <InlineMath math={'\\max(0,z)'} /></p>
                      <p><strong>Pool:</strong> <InlineMath math={'\\max_{2\\times2}(\\cdot)'} /></p>
                    </div>
                  </article>
                </div>
              </div>
            )}
            {stage === 2 && <MatrixCard title={`ReLU channel ${selectedChannel}`} matrix={reluStageMatrix} />}
            {stage === 3 && <MatrixCard title={`Pool channel ${selectedChannel}`} matrix={poolStageMatrix} />}
            {stage === 4 && (
              <article className="deep-matrix-card">
                <p className="deep-mini-title">Flatten preview (first 36)</p>
                <div className="deep-flatten-row">
                  {(isLearnMode ? poolStageMatrix.flat() : inference.flattened).slice(0, 36).map((value, index) => (
                    <span key={`flat-${index}`} className="deep-flat-chip">{value.toFixed(2)}</span>
                  ))}
                </div>
              </article>
            )}
            {stage === 5 && (
              <article className="deep-matrix-card">
                <p className="deep-mini-title">Dense logits (pre-softmax)</p>
                <div className="deep-output-bars">
                  {inference.snapshot.logits.map((value, index) => (
                    <div key={`cnn-logit-${index}`} className="deep-prob-row">
                      <span>{index}</span>
                      <div className="deep-prob-track">
                        <span style={{ width: `${Math.min(100, Math.max(4, (Math.abs(value) / 20) * 100)).toFixed(2)}%` }} />
                      </div>
                      <strong>{value.toFixed(2)}</strong>
                    </div>
                  ))}
                </div>
              </article>
            )}
          </div>

          <article className={`deep-zone-info deep-zone-info-compact ${isPredictCompact ? 'studio-cnn-predict-output-info' : ''}`}>
            <p className="deep-mini-title">Math + Output</p>
            {(!isLearnMode || hasInteracted) && (
              <>
                <p className="deep-feature-copy">Predicted digit: <strong>{inference.snapshot.predictedClass}</strong></p>
                <p className="deep-feature-copy">Top confidence: <strong>{(Math.max(...inference.snapshot.probabilities) * 100).toFixed(2)}%</strong></p>
                <p className="deep-feature-copy">Latency: <strong>{inference.snapshot.latencyMs.toFixed(2)} ms</strong></p>
                <div className="deep-output-bars">
                  {inference.snapshot.probabilities.map((value, index) => (
                    <div key={`cnn-prob-${index}`} className="deep-prob-row">
                      <span>{index}</span>
                      <div className="deep-prob-track"><span style={{ width: `${(value * 100).toFixed(2)}%` }} /></div>
                      <strong>{(value * 100).toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
                {!isLearnMode && (
                  <div className="studio-cnn-predict-summary">
                    <p>
                      Uses trained <strong>mnist_cnn</strong>: logits → softmax probabilities.
                    </p>
                    <p className="studio-cnn-predict-warning">
                      <strong>FYI:</strong> off-center, faint/thick, ambiguous, or out-of-distribution digits can still misclassify.
                    </p>
                  </div>
                )}
              </>
            )}
          </article>
        </div>
      )}

      {inference && isPredictCompact && (
        <div className="studio-cnn-compact-stack">
          <div className="deep-zone-visual deep-zone-visual-large">
            {stage === 0 && <MatrixCard title="Input (28x28)" matrix={image} />}
            {stage === 1 && (
              <div className="deep-cnn-window-grid">
                <MatrixCard title="Input + active patch" matrix={image} highlight={patchHighlight} />
                <MatrixCard title="Kernel (3x3)" matrix={scaledKernel} />
                <div className="studio-cnn-output-cluster">
                  <MatrixCard title="Output feature map (progressive)" matrix={progressiveConv} />
                  <article className="studio-cnn-math-tooltip">
                    <p className="deep-mini-title">Live 3x3 Math</p>
                    <div className="studio-cnn-math-matrix">
                      {matrixTerms.map((row, rowIndex) => (
                        row.map((cell, colIndex) => {
                          const flatIndex = rowIndex * 3 + colIndex;
                          return (
                            <div
                              key={`cnn-math-compact-${rowIndex}-${colIndex}`}
                              className={`studio-cnn-math-cell ${flatIndex === activeTermIndex ? 'is-active' : ''}`}
                            >
                              <span>{cell.pixel.toFixed(2)} x {cell.kernel.toFixed(2)}</span>
                              <strong>{cell.product.toFixed(3)}</strong>
                            </div>
                          );
                        })
                      ))}
                    </div>
                    <p className="studio-cnn-math-sum">
                      Sum = <strong>{convMatrix[scanRow]?.[scanCol]?.toFixed(4) ?? '0.0000'}</strong>
                    </p>
                  </article>
                </div>
              </div>
            )}
            {stage === 2 && <MatrixCard title={`ReLU channel ${selectedChannel}`} matrix={reluStageMatrix} />}
            {stage === 3 && <MatrixCard title={`Pool channel ${selectedChannel}`} matrix={poolStageMatrix} />}
            {stage === 4 && (
              <article className="deep-matrix-card">
                <p className="deep-mini-title">Flatten preview (first 36)</p>
                <div className="deep-flatten-row">
                  {flattenVector.slice(0, 36).map((value, index) => (
                    <span key={`flat-compact-${index}`} className="deep-flat-chip">{value.toFixed(2)}</span>
                  ))}
                </div>
              </article>
            )}
            {stage === 5 && (
              <article className="deep-matrix-card">
                <p className="deep-mini-title">Dense logits (pre-softmax)</p>
                <div className="deep-output-bars">
                  {inference.snapshot.logits.map((value, index) => (
                    <div key={`cnn-logit-compact-${index}`} className="deep-prob-row">
                      <span>{index}</span>
                      <div className="deep-prob-track">
                        <span style={{ width: `${Math.min(100, Math.max(4, (Math.abs(value) / 20) * 100)).toFixed(2)}%` }} />
                      </div>
                      <strong>{value.toFixed(2)}</strong>
                    </div>
                  ))}
                </div>
              </article>
            )}
          </div>

          <div className="studio-cnn-explain-grid">
            {compactLayerTiles.map((tile, index) => (
              <article key={`compact-tile-${tile.title}`} className={`studio-cnn-explain-tile ${stage === index ? 'is-active' : ''}`}>
                <p className="deep-mini-title">{tile.title}</p>
                <p className="deep-feature-copy">{tile.body}</p>
              </article>
            ))}
          </div>
          <p className="studio-cnn-explain-fyi">
            <strong>FYI:</strong> off-center, faint/thick, ambiguous, or out-of-distribution digits can still misclassify.
          </p>
        </div>
      )}
    </section>
  );
}
