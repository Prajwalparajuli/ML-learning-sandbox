import { useEffect, useMemo, useState } from 'react';
import type { DlInferenceMode } from '@/data/deep-learning/types';
import { runCnnInference } from '@/lib/deepLearning/cnnTrace';
import { loadModelAsset, type MnistCnnModel } from '@/lib/deepLearning/modelLoader';
import {
  centerImageByMass,
  normalizeImageMatrix,
  preprocessDrawnDigit,
} from '@/lib/deepLearning/mnistPreprocess';
import { initTfjsRuntime, type TfjsRuntimeStatus } from '@/lib/deepLearning/tfjsRuntime';
import { CnnInferenceExplorer } from './CnnInferenceExplorer';
import { MnistDrawPad } from './MnistDrawPad';
import { TextbookLearnMode } from './TextbookLearnMode';
import './deep-learning-studio.css';

const makeEmptyImage = () => Array.from({ length: 28 }, () => Array.from({ length: 28 }, () => 0));

interface CenterStatus {
  hasSignal: boolean;
  isCentered: boolean;
  offsetX: number;
  offsetY: number;
  distance: number;
}

function measureCentering(image: number[][]): CenterStatus {
  let mass = 0;
  let rowSum = 0;
  let colSum = 0;

  for (let row = 0; row < image.length; row += 1) {
    for (let col = 0; col < image[row].length; col += 1) {
      const value = image[row][col];
      if (value <= 0.06) continue;
      mass += value;
      rowSum += row * value;
      colSum += col * value;
    }
  }

  if (mass <= 1e-6) {
    return {
      hasSignal: false,
      isCentered: true,
      offsetX: 0,
      offsetY: 0,
      distance: 0,
    };
  }

  const target = 13.5;
  const centerRow = rowSum / mass;
  const centerCol = colSum / mass;
  const offsetY = centerRow - target;
  const offsetX = centerCol - target;
  const distance = Math.sqrt(offsetX * offsetX + offsetY * offsetY);

  return {
    hasSignal: true,
    isCentered: distance <= 2.35,
    offsetX,
    offsetY,
    distance,
  };
}

export function DeepLearningPanel() {
  const [studioMode, setStudioMode] = useState<'predict' | 'learn'>('predict');
  const inferenceMode: DlInferenceMode = 'pure';
  const [image, setImage] = useState<number[][]>(makeEmptyImage);
  const [predictionImage, setPredictionImage] = useState<number[][]>(makeEmptyImage);
  const [predictNonce, setPredictNonce] = useState(0);
  const [runtime, setRuntime] = useState<TfjsRuntimeStatus | null>(null);
  const [cnnModel, setCnnModel] = useState<MnistCnnModel | null>(null);
  const [centerPopupOpen, setCenterPopupOpen] = useState(false);

  useEffect(() => {
    let active = true;
    initTfjsRuntime().then((status) => {
      if (active) setRuntime(status);
    });
    loadModelAsset('mnist_cnn').then((model) => {
      if (active) setCnnModel(model as MnistCnnModel);
    });
    return () => {
      active = false;
    };
  }, []);

  const drawCenterStatus = useMemo(
    () => measureCentering(normalizeImageMatrix(image)),
    [image]
  );
  const hasPredictionSignal = useMemo(
    () => predictionImage.some((row) => row.some((value) => value > 0.06)),
    [predictionImage]
  );
  const cnnInference = useMemo(() => {
    if (!hasPredictionSignal || !cnnModel) return null;
    return runCnnInference(predictionImage, cnnModel, {
      topChannels: 4,
      inferenceMode,
    });
  }, [cnnModel, hasPredictionSignal, inferenceMode, predictionImage]);
  const topPredictions = useMemo(() => {
    if (!cnnInference) return [];
    return cnnInference.snapshot.probabilities
      .map((value, digit) => ({ digit, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 3);
  }, [cnnInference]);

  const handleImageChange = (nextImage: number[][]) => {
    setImage(nextImage);
    setPredictionImage(makeEmptyImage());
    if (centerPopupOpen) setCenterPopupOpen(false);
  };

  const commitPrediction = (sourceImage: number[][]) => {
    const normalized = normalizeImageMatrix(sourceImage);
    setPredictionImage(preprocessDrawnDigit(normalized));
    setPredictNonce((value) => value + 1);
  };

  const triggerPredict = () => {
    if (!drawCenterStatus.hasSignal) return;
    if (!drawCenterStatus.isCentered) {
      setCenterPopupOpen(true);
      return;
    }
    commitPrediction(image);
  };

  const autoCenterAndPredict = () => {
    const centered = centerImageByMass(normalizeImageMatrix(image));
    setImage(centered);
    setCenterPopupOpen(false);
    commitPrediction(centered);
  };

  if (studioMode === 'learn') {
    return (
      <div className="studio-shell">
        <section className="studio-hero">
          <div className="studio-hero-copy">
            <h2 className="studio-title">Deep Learning Studio</h2>
          </div>
          <div className="studio-hero-tools">
            <div className="studio-runtime">
              <div className="studio-runtime-pill">
                Runtime <strong>{runtime?.backend ?? 'initializing'}</strong>
              </div>
              <div className="studio-runtime-pill">
                Status <strong>{runtime?.ready ? 'ready' : 'starting'}</strong>
              </div>
              <div className="studio-runtime-pill">
                Engine <strong>{runtime?.usingTfjs ? 'tfjs' : 'numeric fallback'}</strong>
              </div>
            </div>
            <div className="studio-mode-toggle" role="group" aria-label="Studio mode">
              <button type="button" className="studio-mode-chip" onClick={() => setStudioMode('predict')}>
                Predict mode (CNN only)
              </button>
              <button type="button" className="studio-mode-chip is-active" onClick={() => setStudioMode('learn')}>
                Learn mode (Textbook)
              </button>
            </div>
          </div>
        </section>
        <TextbookLearnMode />
      </div>
    );
  }

  return (
    <div className="studio-shell">
      {centerPopupOpen && (
        <div className="studio-modal-backdrop" role="dialog" aria-modal="true" aria-label="Center digit before prediction">
          <div className="studio-modal-card">
            <h4>Center the digit before prediction</h4>
            <p>
              The digit centroid is off-center by <strong>{drawCenterStatus.distance.toFixed(2)} px</strong>
              {' '}({drawCenterStatus.offsetX.toFixed(1)} px x, {drawCenterStatus.offsetY.toFixed(1)} px y).
            </p>
            <p>Centering improves consistency, so prediction is blocked until alignment is fixed.</p>
            <div className="studio-modal-actions">
              <button type="button" className="studio-flow-model is-active" onClick={autoCenterAndPredict}>
                Auto-center + Predict
              </button>
              <button type="button" className="studio-flow-model" onClick={() => setCenterPopupOpen(false)}>
                I will adjust manually
              </button>
            </div>
          </div>
        </div>
      )}

      <section className="studio-hero">
        <div className="studio-hero-copy">
          <h2 className="studio-title">Deep Learning Studio</h2>
          <p className="studio-subtitle">CNN Predict + Learn</p>
        </div>
        <div className="studio-hero-tools">
          <div className="studio-runtime">
            <div className="studio-runtime-pill">Runtime <strong>{runtime?.backend ?? 'initializing'}</strong></div>
            <div className="studio-runtime-pill">Status <strong>{runtime?.ready ? 'ready' : 'starting'}</strong></div>
            <div className="studio-runtime-pill">Engine <strong>{runtime?.usingTfjs ? 'tfjs' : 'numeric fallback'}</strong></div>
            {cnnInference && (
              <div className="studio-runtime-pill studio-runtime-pill-optional">
                CNN <strong>{cnnInference.snapshot.predictedClass}</strong> (
                {((cnnInference.snapshot.probabilities[cnnInference.snapshot.predictedClass] ?? 0) * 100).toFixed(1)}%)
              </div>
            )}
          </div>
          <div className="studio-mode-toggle" role="group" aria-label="Studio mode">
            <button type="button" className="studio-mode-chip is-active" onClick={() => setStudioMode('predict')}>
              Predict mode (CNN only)
            </button>
            <button type="button" className="studio-mode-chip" onClick={() => setStudioMode('learn')}>
              Learn mode (Textbook)
            </button>
          </div>
        </div>
      </section>

      <section className="studio-card studio-predict-card">
        <div className="studio-predict-layout">
          <div className="studio-pad-wrap studio-predict-pad">
            <MnistDrawPad
              image={image}
              onImageChange={handleImageChange}
              onPredict={triggerPredict}
              centerStatus={drawCenterStatus}
              predictionSummary={null}
            />
            <article className="studio-predict-output-tile">
              <p className="studio-mini-title">Prediction Output</p>
              {!cnnInference && (
                <p className="deep-feature-copy">
                  Draw and click <strong>Predict</strong>.
                </p>
              )}
              {cnnInference && (
                <>
                  <p className="deep-feature-copy">
                    Predicted digit <strong>{cnnInference.snapshot.predictedClass}</strong> at{' '}
                    <strong>{((cnnInference.snapshot.probabilities[cnnInference.snapshot.predictedClass] ?? 0) * 100).toFixed(1)}%</strong>.
                    {' '}Latency <strong>{cnnInference.snapshot.latencyMs.toFixed(2)} ms</strong>.
                  </p>
                  <div className="studio-bar-list studio-predict-output-bars">
                    {topPredictions.map((entry, index) => (
                      <div key={`predict-top-${entry.digit}`} className="studio-bar-row">
                        <span>{entry.digit}</span>
                        <div className="studio-bar-track">
                          <span className={index === 0 ? 'is-active' : ''} style={{ width: `${(entry.value * 100).toFixed(2)}%` }} />
                        </div>
                        <strong>{(entry.value * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </article>
          </div>
          <div className="studio-cnn-live-wrap studio-cnn-live-wrap-compact">
            <CnnInferenceExplorer
              image={predictionImage}
              model={cnnModel}
              topChannels={4}
              predictNonce={predictNonce}
              inferenceMode={inferenceMode}
              experienceMode="predict"
              compactPredict
            />
          </div>
        </div>
      </section>
    </div>
  );
}
