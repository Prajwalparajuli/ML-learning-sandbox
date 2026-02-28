import type {
  CnnActivationTrace,
  CnnMapSlice,
  DlInferenceMode,
  InferenceSnapshot,
} from '@/data/deep-learning/types';
import type { MnistCnnModel } from './modelLoader';
import { extractCnnFeatures } from './cnnFeatures';
import { logitsFromPrototypes } from './digitPrototypes';
import { logitsFromKnnMemory } from './digitMemory';
import { argMax, convValid, flattenImageMatrix, maxPool2x2, reluMatrix, softmax, topKIndices } from './mnistPreprocess';

export interface CnnInferenceResult {
  snapshot: InferenceSnapshot;
  trace: CnnActivationTrace;
  flattened: number[];
}

interface CnnTraceOptions {
  topChannels?: number;
  inferenceMode?: DlInferenceMode;
}

const meanOfMatrix = (matrix: number[][]): number => {
  let sum = 0;
  let count = 0;
  for (let r = 0; r < matrix.length; r += 1) {
    for (let c = 0; c < matrix[r].length; c += 1) {
      sum += matrix[r][c];
      count += 1;
    }
  }
  return count > 0 ? sum / count : 0;
};

const toSlice = (channel: number, values: number[][]): CnnMapSlice => ({
  channel,
  values,
  meanActivation: meanOfMatrix(values),
});

const dot = (a: number[], b: number[]): number => {
  let sum = 0;
  const limit = Math.min(a.length, b.length);
  for (let i = 0; i < limit; i += 1) sum += a[i] * b[i];
  return sum;
};

const extraConvKernel = [
  [0.05, 0.10, 0.05],
  [0.10, 0.40, 0.10],
  [0.05, 0.10, 0.05],
];

export function runCnnInference(
  image: number[][],
  model: MnistCnnModel,
  options: CnnTraceOptions = {}
): CnnInferenceResult {
  const t0 = performance.now();
  const inferenceMode: DlInferenceMode = options.inferenceMode ?? 'assisted';
  const features = extractCnnFeatures(image, model);
  const { centered, convMaps, reluMaps, poolMaps, flattened } = features;
  const inputFlat = flattenImageMatrix(centered);
  const baseLogits = model.denseW.length > 0 && model.denseW[0]?.length === flattened.length
    ? model.denseW.map(
      (weights, classIndex) => dot(weights, flattened) + (model.denseB[classIndex] ?? 0)
    )
    : logitsFromPrototypes(inputFlat, {
      mseWeight: 1.37,
      scale: 20,
    });

  const logits = baseLogits.slice();

  if (inferenceMode === 'assisted') {
    const prototypeLogits = logitsFromPrototypes(inputFlat, {
      mseWeight: 0.85,
      scale: 18,
    });
    const knnLogits = logitsFromKnnMemory(inputFlat, {
      k: 29,
      temperature: 22,
      scale: 20,
    });

    // Extra conv layer: depthwise conv over ReLU feature maps -> ReLU -> pool -> channel energies.
    const conv2Maps = reluMaps.map((map) => reluMatrix(convValid(map, extraConvKernel)));
    const pool2Maps = conv2Maps.map((map) => maxPool2x2(map));
    const channelEnergies = pool2Maps.map((map) => meanOfMatrix(map));

    // Map conv2 channel energies into class logits by reusing class/channel weight trends.
    const conv2Logits = Array.from({ length: model.denseW.length }, (_, classIndex) => {
      const weights = model.denseW[classIndex] ?? [];
      let z = model.denseB[classIndex] ?? 0;
      for (let channel = 0; channel < channelEnergies.length; channel += 1) {
        const start = channel * 13 * 13;
        const end = Math.min(weights.length, start + 13 * 13);
        if (end <= start) continue;
        let meanWeight = 0;
        for (let i = start; i < end; i += 1) meanWeight += weights[i];
        meanWeight /= Math.max(1, end - start);
        z += meanWeight * channelEnergies[channel] * 28;
      }
      return z;
    });

    for (let classIndex = 0; classIndex < model.denseW.length; classIndex += 1) {
      logits[classIndex] =
        (baseLogits[classIndex] ?? 0) * 0.10 +
        (prototypeLogits[classIndex] ?? 0) * 0.23 +
        (conv2Logits[classIndex] ?? 0) * 0.07 +
        (knnLogits[classIndex] ?? 0) * 0.60;
    }

    const firstPass = softmax(logits);
    const ranked = firstPass
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value);
    const top1 = ranked[0]?.value ?? 0;
    const top2 = ranked[1]?.value ?? 0;
    const lowConfidence = top1 < 0.72 || top1 - top2 < 0.18;
    if (lowConfidence) {
      for (let i = 0; i < logits.length; i += 1) {
        logits[i] = logits[i] * 0.22 + knnLogits[i] * 0.78;
      }
    }
  }

  const probabilities = softmax(logits);
  const predictedClass = argMax(probabilities);

  const meanScores = reluMaps.map((map) => meanOfMatrix(map));
  const topChannels = topKIndices(meanScores, Math.min(options.topChannels ?? 4, reluMaps.length));
  const convSlices = topChannels.map((index) => toSlice(index, convMaps[index]));
  const reluSlices = topChannels.map((index) => toSlice(index, reluMaps[index]));
  const poolSlices = topChannels.map((index) => toSlice(index, poolMaps[index]));

  const snapshot: InferenceSnapshot = {
    logits,
    probabilities,
    predictedClass,
    latencyMs: Number((performance.now() - t0).toFixed(3)),
  };

  const trace: CnnActivationTrace = {
    conv: convSlices,
    relu: reluSlices,
    pool: poolSlices,
    probabilities,
  };

  return {
    snapshot,
    trace,
    flattened,
  };
}
