import type { MnistCnnModel } from './modelLoader';
import { centerImageByMass, convValid, flattenImageMatrix, maxPool2x2, reluMatrix } from './mnistPreprocess';

export interface CnnFeatureExtraction {
  centered: number[][];
  convMaps: number[][][];
  reluMaps: number[][][];
  poolMaps: number[][][];
  flattened: number[];
}

export function extractCnnFeatures(image: number[][], model: MnistCnnModel): CnnFeatureExtraction {
  const centered = centerImageByMass(image);
  const convMaps = model.kernels.map((kernel) => convValid(centered, kernel));
  const reluMaps = convMaps.map((map) => reluMatrix(map));
  const poolMaps = reluMaps.map((map) => maxPool2x2(map));
  const flattened = poolMaps.flatMap((map) => flattenImageMatrix(map));
  return { centered, convMaps, reluMaps, poolMaps, flattened };
}
