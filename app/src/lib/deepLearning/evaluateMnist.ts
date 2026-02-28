import type { DlEvalMetrics, DlInferenceMode } from '@/data/deep-learning/types';
import evalSource from '@/data/deep-learning/eval/mnist_eval_1000.json';
import { runCnnInference } from './cnnTrace';
import { runMlpInference } from './mlpTrace';
import type { MnistCnnModel, MnistMlpModel } from './modelLoader';
import { decodeBase64ToU8, matrixFromPackedU8 } from './mnistPreprocess';

interface MnistEvalPayload {
  version: number;
  count: number;
  image_shape: [number, number, number];
  images_u8_b64: string;
  labels: number[];
}

interface EvaluateOptions {
  batchSize?: number;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
  inferenceMode?: DlInferenceMode;
}

type EvaluateKind = 'mnist_mlp' | 'mnist_cnn';

const evalPayload = evalSource as MnistEvalPayload;
const metricsCache = new Map<string, DlEvalMetrics>();
let packedImages: Uint8Array | null = null;

const yieldToMain = () =>
  new Promise<void>((resolve) => {
    if (typeof window !== 'undefined') {
      const win = window as Window & {
        requestIdleCallback?: (callback: (deadline: IdleDeadline) => void) => number;
      };
      if (typeof win.requestIdleCallback === 'function') {
        win.requestIdleCallback(() => resolve());
        return;
      }
    }
    setTimeout(resolve, 0);
  });

const getPackedImages = (): Uint8Array => {
  if (!packedImages) packedImages = decodeBase64ToU8(evalPayload.images_u8_b64);
  return packedImages;
};

const computeMacroPrecision = (confusion: number[][]): number => {
  let sum = 0;
  for (let c = 0; c < 10; c += 1) {
    const tp = confusion[c][c];
    let fp = 0;
    for (let r = 0; r < 10; r += 1) {
      if (r !== c) fp += confusion[r][c];
    }
    sum += tp / Math.max(tp + fp, Number.EPSILON);
  }
  return sum / 10;
};

const computeMacroRecall = (confusion: number[][]): number => {
  let sum = 0;
  for (let c = 0; c < 10; c += 1) {
    const tp = confusion[c][c];
    let fn = 0;
    for (let p = 0; p < 10; p += 1) {
      if (p !== c) fn += confusion[c][p];
    }
    sum += tp / Math.max(tp + fn, Number.EPSILON);
  }
  return sum / 10;
};

export async function evaluateMnistModel(
  kind: 'mnist_mlp',
  model: MnistMlpModel,
  options?: EvaluateOptions
): Promise<DlEvalMetrics>;
export async function evaluateMnistModel(
  kind: 'mnist_cnn',
  model: MnistCnnModel,
  options?: EvaluateOptions
): Promise<DlEvalMetrics>;
export async function evaluateMnistModel(
  kind: EvaluateKind,
  model: MnistMlpModel | MnistCnnModel,
  options: EvaluateOptions = {}
): Promise<DlEvalMetrics> {
  const inferenceMode: DlInferenceMode = options.inferenceMode ?? 'assisted';
  const cacheKey = `${kind}:${inferenceMode}`;
  const cached = metricsCache.get(cacheKey);
  if (cached) return cached;

  const packed = getPackedImages();
  const confusion = Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => 0));
  const batchSize = Math.max(1, options.batchSize ?? 64);
  let correct = 0;

  for (let start = 0; start < evalPayload.count; start += batchSize) {
    if (options.signal?.aborted) throw new Error('Evaluation aborted');
    const end = Math.min(evalPayload.count, start + batchSize);
    for (let i = start; i < end; i += 1) {
      const image = matrixFromPackedU8(packed, i, evalPayload.image_shape);
      const label = evalPayload.labels[i];
      const predictedClass = kind === 'mnist_mlp'
        ? runMlpInference(image, model as MnistMlpModel, { inferenceMode }).snapshot.predictedClass
        : runCnnInference(image, model as MnistCnnModel, { inferenceMode }).snapshot.predictedClass;
      confusion[label][predictedClass] += 1;
      if (predictedClass === label) correct += 1;
    }
    options.onProgress?.(end, evalPayload.count);
    await yieldToMain();
  }

  const metrics: DlEvalMetrics = {
    accuracy: correct / evalPayload.count,
    precisionMacro: computeMacroPrecision(confusion),
    recallMacro: computeMacroRecall(confusion),
    confusion10x10: confusion,
    sampleCount: evalPayload.count,
  };

  metricsCache.set(cacheKey, metrics);
  return metrics;
}
