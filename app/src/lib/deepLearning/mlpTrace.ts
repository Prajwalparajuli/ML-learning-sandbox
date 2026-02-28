import type {
  DlInferenceMode,
  InferenceSnapshot,
  MlpActivationTrace,
  MlpContributionEdge,
} from '@/data/deep-learning/types';
import type { MnistMlpModel } from './modelLoader';
import { logitsFromPrototypes } from './digitPrototypes';
import { logitsFromKnnMemory } from './digitMemory';
import { argMax, centerImageByMass, flattenImageMatrix, groupInto16, softmax, topKIndices } from './mnistPreprocess';

export interface MlpInferenceResult {
  snapshot: InferenceSnapshot;
  trace: MlpActivationTrace;
}

export interface MlpTraceOptions {
  topK?: number;
  hiddenCap?: number;
  groupEdgesPerHidden?: number;
  inferenceMode?: DlInferenceMode;
}

const defaultOptions: Required<MlpTraceOptions> = {
  topK: 12,
  hiddenCap: 24,
  groupEdgesPerHidden: 2,
  inferenceMode: 'assisted',
};

const buildGroupIndexMap = (): number[][] => {
  const groups = Array.from({ length: 16 }, () => [] as number[]);
  for (let r = 0; r < 28; r += 1) {
    for (let c = 0; c < 28; c += 1) {
      const gr = Math.min(3, Math.floor(r / 7));
      const gc = Math.min(3, Math.floor(c / 7));
      groups[gr * 4 + gc].push(r * 28 + c);
    }
  }
  return groups;
};

const groupIndexMap = buildGroupIndexMap();

const dot = (a: number[], b: number[]): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * (b[i] ?? 0);
  return sum;
};

export function runMlpInference(
  image: number[][],
  model: MnistMlpModel,
  options: MlpTraceOptions = {}
): MlpInferenceResult {
  const config = { ...defaultOptions, ...options };
  const t0 = performance.now();

  const centered = centerImageByMass(image);
  const input = flattenImageMatrix(centered);
  const inferenceMode: DlInferenceMode = options.inferenceMode ?? 'assisted';
  const hiddenZ = Array.from({ length: model.hiddenSize }, () => 0);
  const hiddenA = Array.from({ length: model.hiddenSize }, () => 0);

  for (let h = 0; h < model.hiddenSize; h += 1) {
    const z = dot(model.w1[h], input) + model.b1[h];
    hiddenZ[h] = z;
    hiddenA[h] = z > 0 ? z : 0;
  }

  const baseLogits = Array.from({ length: model.outputSize }, (_, outputIndex) => (
    dot(model.w2[outputIndex], hiddenA) + (model.b2[outputIndex] ?? 0)
  ));
  const logits = baseLogits.slice();
  if (inferenceMode === 'assisted') {
    const prototypeLogits = logitsFromPrototypes(input, { mseWeight: 0.85, scale: 18 });
    const knnLogits = logitsFromKnnMemory(input, {
      k: 25,
      temperature: 20,
      scale: 18,
    });

    for (let outputIndex = 0; outputIndex < model.outputSize; outputIndex += 1) {
      logits[outputIndex] =
        baseLogits[outputIndex] * 0.24 +
        prototypeLogits[outputIndex] * 0.34 +
        knnLogits[outputIndex] * 0.42;
    }

    const firstPass = softmax(logits);
    const ranked = firstPass
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value);
    const top1 = ranked[0]?.value ?? 0;
    const top2 = ranked[1]?.value ?? 0;
    const lowConfidence = top1 < 0.65 || top1 - top2 < 0.15;
    if (lowConfidence) {
      for (let i = 0; i < logits.length; i += 1) {
        logits[i] = logits[i] * 0.25 + knnLogits[i] * 0.75;
      }
    }
  }

  const probabilities = softmax(logits);
  const predictedClass = argMax(probabilities);

  const predictedWeights = model.w2[predictedClass];
  const hiddenContributionScores = hiddenA.map((value, index) => value * predictedWeights[index]);
  const selectedHidden = topKIndices(
    hiddenContributionScores,
    Math.min(config.hiddenCap, Math.max(1, config.topK))
  );

  const edges: MlpContributionEdge[] = [];

  const hiddenToOutputCandidates: Array<{ hiddenIndex: number; contribution: number }> = [];
  for (let h = 0; h < selectedHidden.length; h += 1) {
    const hiddenIndex = selectedHidden[h];
    hiddenToOutputCandidates.push({
      hiddenIndex,
      contribution: hiddenA[hiddenIndex] * model.w2[predictedClass][hiddenIndex],
    });
  }
  hiddenToOutputCandidates
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, Math.max(1, Math.min(config.topK, selectedHidden.length)))
    .forEach((candidate) => {
      edges.push({
        sourceLayer: 'hidden',
        sourceIndex: candidate.hiddenIndex,
        targetLayer: 'output',
        targetIndex: predictedClass,
        contribution: candidate.contribution,
      });
    });

  for (let i = 0; i < selectedHidden.length; i += 1) {
    const hiddenIndex = selectedHidden[i];
    const groupScores = groupIndexMap.map((pixelIndices) => {
      let sum = 0;
      for (let p = 0; p < pixelIndices.length; p += 1) {
        const pixelIndex = pixelIndices[p];
        sum += input[pixelIndex] * model.w1[hiddenIndex][pixelIndex];
      }
      return sum;
    });
    const topGroups = topKIndices(groupScores, config.groupEdgesPerHidden);
    for (let g = 0; g < topGroups.length; g += 1) {
      const groupIndex = topGroups[g];
      edges.push({
        sourceLayer: 'input_group',
        sourceIndex: groupIndex,
        targetLayer: 'hidden',
        targetIndex: hiddenIndex,
        contribution: groupScores[groupIndex],
      });
    }
  }

  const snapshot: InferenceSnapshot = {
    logits,
    probabilities,
    predictedClass,
    latencyMs: Number((performance.now() - t0).toFixed(3)),
  };

  const trace: MlpActivationTrace = {
    inputGroups: groupInto16(centered),
    hiddenActivations: hiddenA,
    outputLogits: logits,
    contributionEdges: edges,
  };

  return { snapshot, trace };
}
