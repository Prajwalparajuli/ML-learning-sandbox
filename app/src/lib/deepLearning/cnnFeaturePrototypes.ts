import evalSource from '@/data/deep-learning/eval/mnist_eval_1000.json';
import type { MnistCnnModel } from './modelLoader';
import { decodeBase64ToU8, matrixFromPackedU8 } from './mnistPreprocess';
import { extractCnnFeatures } from './cnnFeatures';

interface MnistEvalPayload {
  version: number;
  count: number;
  image_shape: [number, number, number];
  images_u8_b64: string;
  labels: number[];
}

const payload = evalSource as MnistEvalPayload;

interface PrototypeCacheRecord {
  modelKey: string;
  prototypes: number[][];
}

let packedImages: Uint8Array | null = null;
let prototypeCache: PrototypeCacheRecord | null = null;

const dot = (a: number[], b: number[]): number => {
  let sum = 0;
  const limit = Math.min(a.length, b.length);
  for (let i = 0; i < limit; i += 1) sum += a[i] * b[i];
  return sum;
};

const norm = (values: number[]): number => Math.sqrt(dot(values, values));

const getPackedImages = (): Uint8Array => {
  if (!packedImages) packedImages = decodeBase64ToU8(payload.images_u8_b64);
  return packedImages;
};

const modelKeyFromKernels = (model: MnistCnnModel): string => {
  const samples = model.kernels.flatMap((kernel) => kernel.flat()).slice(0, 12).join('|');
  return `${model.kernels.length}:${samples}`;
};

function buildPrototypes(model: MnistCnnModel): number[][] {
  const packed = getPackedImages();
  const featureLength = model.denseW[0]?.length ?? 1352;
  const sums = Array.from({ length: 10 }, () => Array.from({ length: featureLength }, () => 0));
  const counts = Array.from({ length: 10 }, () => 0);

  for (let i = 0; i < payload.count; i += 1) {
    const label = payload.labels[i];
    const image = matrixFromPackedU8(packed, i, payload.image_shape);
    const feature = extractCnnFeatures(image, model).flattened;
    counts[label] += 1;
    for (let p = 0; p < featureLength; p += 1) {
      sums[label][p] += feature[p] ?? 0;
    }
  }

  return sums.map((row, label) => row.map((value) => value / Math.max(1, counts[label])));
}

function getPrototypes(model: MnistCnnModel): number[][] {
  const modelKey = modelKeyFromKernels(model);
  if (!prototypeCache || prototypeCache.modelKey !== modelKey) {
    prototypeCache = {
      modelKey,
      prototypes: buildPrototypes(model),
    };
  }
  return prototypeCache.prototypes;
}

export function logitsFromCnnFeaturePrototypes(feature: number[], model: MnistCnnModel): number[] {
  const prototypes = getPrototypes(model);
  const featureNorm = norm(feature);
  return prototypes.map((prototype) => {
    const protoNorm = norm(prototype);
    const cosine = dot(feature, prototype) / Math.max(featureNorm * protoNorm, 1e-6);
    let mse = 0;
    const limit = Math.min(feature.length, prototype.length);
    for (let i = 0; i < limit; i += 1) {
      const diff = feature[i] - prototype[i];
      mse += diff * diff;
    }
    mse /= Math.max(1, limit);
    const score = cosine - (0.95 * mse);
    return score * 22;
  });
}
