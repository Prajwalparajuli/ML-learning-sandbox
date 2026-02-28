import type { DeepModelKind } from '@/data/deep-learning/types';
import boundaryDeepMeta from '@/data/deep-learning/models/boundary/deep/model.json';
import boundaryShallowMeta from '@/data/deep-learning/models/boundary/shallow/model.json';
import cnnMeta from '@/data/deep-learning/models/mnist/cnn/model.json';
import cnnWeights from '@/data/deep-learning/models/mnist/cnn/weights-shard1.json';
import mlpMeta from '@/data/deep-learning/models/mnist/mlp/model.json';
import mlpWeights from '@/data/deep-learning/models/mnist/mlp/weights-shard1.json';

export interface MnistMlpModel {
  kind: 'mnist_mlp';
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  activation: 'relu';
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
}

export interface MnistCnnModel {
  kind: 'mnist_cnn';
  inputShape: [number, number, number];
  kernels: number[][][];
  denseW: number[][];
  denseB: number[];
}

export interface BoundaryShallowModel {
  kind: 'boundary_shallow';
  coefficients: {
    wx: number;
    wy: number;
    wxy: number;
    bias: number;
  };
}

export interface BoundaryDeepModel {
  kind: 'boundary_deep';
  coefficients: {
    r_center: number;
    r_scale: number;
    swirl: number;
    x_shift: number;
    y_shift: number;
  };
}

export type LoadedModelAsset =
  | MnistMlpModel
  | MnistCnnModel
  | BoundaryShallowModel
  | BoundaryDeepModel;

const cache = new Map<DeepModelKind, LoadedModelAsset>();

const toMlpModel = (): MnistMlpModel => {
  return {
    kind: 'mnist_mlp',
    inputSize: mlpMeta.input_size,
    hiddenSize: mlpMeta.hidden_size,
    outputSize: mlpMeta.output_size,
    activation: 'relu',
    w1: mlpWeights.w1,
    b1: mlpWeights.b1,
    w2: mlpWeights.w2,
    b2: mlpWeights.b2,
  };
};

const toCnnModel = (): MnistCnnModel => {
  return {
    kind: 'mnist_cnn',
    inputShape: [
      cnnMeta.input_size[0],
      cnnMeta.input_size[1],
      cnnMeta.input_size[2],
    ],
    kernels: cnnWeights.kernels,
    denseW: cnnWeights.dense_w,
    denseB: cnnWeights.dense_b,
  };
};

const toBoundaryShallow = (): BoundaryShallowModel => {
  return {
    kind: 'boundary_shallow',
    coefficients: {
      wx: boundaryShallowMeta.coefficients.wx,
      wy: boundaryShallowMeta.coefficients.wy,
      wxy: boundaryShallowMeta.coefficients.wxy,
      bias: boundaryShallowMeta.coefficients.bias,
    },
  };
};

const toBoundaryDeep = (): BoundaryDeepModel => {
  return {
    kind: 'boundary_deep',
    coefficients: {
      r_center: boundaryDeepMeta.coefficients.r_center,
      r_scale: boundaryDeepMeta.coefficients.r_scale,
      swirl: boundaryDeepMeta.coefficients.swirl,
      x_shift: boundaryDeepMeta.coefficients.x_shift,
      y_shift: boundaryDeepMeta.coefficients.y_shift,
    },
  };
};

export async function loadModelAsset(kind: DeepModelKind): Promise<LoadedModelAsset> {
  const hit = cache.get(kind);
  if (hit) return hit;

  let model: LoadedModelAsset;
  if (kind === 'mnist_mlp') {
    model = toMlpModel();
  } else if (kind === 'mnist_cnn') {
    model = toCnnModel();
  } else if (kind === 'boundary_shallow') {
    model = toBoundaryShallow();
  } else {
    model = toBoundaryDeep();
  }

  cache.set(kind, model);
  return model;
}

