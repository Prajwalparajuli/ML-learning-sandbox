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

interface MemoryCache {
  vectors: number[][];
  labels: number[];
  norms: number[];
  evenIndices: number[];
}

interface MemoryMatch {
  label: number;
  similarity: number;
}

interface KnnOptions {
  k?: number;
  temperature?: number;
  scale?: number;
}

const payload = evalSource as MnistEvalPayload;
let memoryCache: MemoryCache | null = null;

const dot = (a: number[], b: number[]): number => {
  let sum = 0;
  const limit = Math.min(a.length, b.length);
  for (let i = 0; i < limit; i += 1) sum += a[i] * b[i];
  return sum;
};

const norm = (values: number[]): number => Math.sqrt(dot(values, values));

function getMemory(): MemoryCache {
  if (memoryCache) return memoryCache;

  const packed = decodeBase64ToU8(payload.images_u8_b64);
  const vectors: number[][] = [];
  const labels = payload.labels.slice();
  const norms: number[] = [];
  const evenIndices: number[] = [];

  for (let i = 0; i < payload.count; i += 1) {
    const image = matrixFromPackedU8(packed, i, payload.image_shape);
    const centered = centerImageByMass(image);
    const vector = flattenImageMatrix(centered);
    vectors.push(vector);
    norms.push(Math.max(1e-6, norm(vector)));
    if ((labels[i] ?? 1) % 2 === 0) evenIndices.push(i);
  }

  memoryCache = {
    vectors,
    labels,
    norms,
    evenIndices,
  };
  return memoryCache;
}

export function nearestEvenDigit(inputFlat: number[]): MemoryMatch | null {
  const memory = getMemory();
  const inputNorm = Math.max(1e-6, norm(inputFlat));

  let bestSimilarity = Number.NEGATIVE_INFINITY;
  let bestLabel = -1;
  for (let i = 0; i < memory.evenIndices.length; i += 1) {
    const index = memory.evenIndices[i];
    const similarity =
      dot(inputFlat, memory.vectors[index]) /
      Math.max(1e-6, inputNorm * memory.norms[index]);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestLabel = memory.labels[index];
    }
  }

  if (bestLabel < 0) return null;
  return {
    label: bestLabel,
    similarity: bestSimilarity,
  };
}

export function logitsFromKnnMemory(inputFlat: number[], options: KnnOptions = {}): number[] {
  const memory = getMemory();
  const k = Math.max(3, Math.min(memory.vectors.length, options.k ?? 21));
  const temperature = Math.max(1, options.temperature ?? 20);
  const scale = Math.max(1, options.scale ?? 18);
  const inputNorm = Math.max(1e-6, norm(inputFlat));

  const neighbors = memory.vectors
    .map((vector, index) => {
      const similarity =
        dot(inputFlat, vector) /
        Math.max(1e-6, inputNorm * memory.norms[index]);
      return {
        similarity,
        label: memory.labels[index],
      };
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);

  const maxSimilarity = neighbors[0]?.similarity ?? 0;
  const scores = Array.from({ length: 10 }, () => 1e-6);
  for (let i = 0; i < neighbors.length; i += 1) {
    const neighbor = neighbors[i];
    const weight = Math.exp((neighbor.similarity - maxSimilarity) * temperature);
    scores[neighbor.label] += weight;
  }

  const total = scores.reduce((sum, value) => sum + value, 0);
  const probabilities = scores.map((value) => value / Math.max(total, Number.EPSILON));
  return probabilities.map((value) => Math.log(Math.max(1e-8, value)) * scale);
}
