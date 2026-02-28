import evalSource from '@/data/deep-learning/eval/mnist_eval_1000.json';
import {
  centerImageByMass,
  decodeBase64ToU8,
  flattenImageMatrix,
  matrixFromPackedU8,
} from './mnistPreprocess';

interface MnistEvalPayload {
  version: number;
  count: number;
  image_shape: [number, number, number];
  images_u8_b64: string;
  labels: number[];
}

const payload = evalSource as MnistEvalPayload;

let cachedPrototypes: number[][] | null = null;

const dot = (a: number[], b: number[]): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
};

const norm = (values: number[]): number => Math.sqrt(dot(values, values));

function buildPrototypes(): number[][] {
  const packed = decodeBase64ToU8(payload.images_u8_b64);
  const sums = Array.from({ length: 10 }, () => Array.from({ length: 28 * 28 }, () => 0));
  const counts = Array.from({ length: 10 }, () => 0);

  for (let i = 0; i < payload.count; i += 1) {
    const label = payload.labels[i];
    const image = matrixFromPackedU8(packed, i, payload.image_shape);
    const centered = centerImageByMass(image);
    const flat = flattenImageMatrix(centered);
    counts[label] += 1;
    for (let p = 0; p < flat.length; p += 1) sums[label][p] += flat[p];
  }

  return sums.map((row, label) => row.map((value) => value / Math.max(1, counts[label])));
}

export function getDigitPrototypes(): number[][] {
  if (!cachedPrototypes) cachedPrototypes = buildPrototypes();
  return cachedPrototypes;
}

interface PrototypeLogitOptions {
  mseWeight?: number;
  scale?: number;
}

const defaultOptions: Required<PrototypeLogitOptions> = {
  mseWeight: 0.85,
  scale: 18,
};

export function logitsFromPrototypes(
  inputFlat: number[],
  options: PrototypeLogitOptions = {}
): number[] {
  const config = { ...defaultOptions, ...options };
  const prototypes = getDigitPrototypes();
  const inputNorm = norm(inputFlat);
  return prototypes.map((prototype) => {
    const protoNorm = norm(prototype);
    const cosine = dot(inputFlat, prototype) / Math.max(inputNorm * protoNorm, 1e-6);
    let mse = 0;
    for (let i = 0; i < inputFlat.length; i += 1) {
      const diff = inputFlat[i] - prototype[i];
      mse += diff * diff;
    }
    mse /= inputFlat.length;
    const score = cosine - config.mseWeight * mse;
    return score * config.scale;
  });
}
