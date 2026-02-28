import type { CnnStage } from './godModeTypes';

export const MLP_X: [number, number] = [0.5, 1.0];
export const MLP_INITIAL_W: [number, number] = [0.8, -0.4];
export const MLP_BIAS = -0.2;
export const MLP_TARGET = 1.0;

export const CNN_INPUT: number[][] = [
  [0.0, 0.1, 0.9, 0.9, 0.1, 0.0],
  [0.0, 0.1, 1.0, 1.0, 0.1, 0.0],
  [0.0, 0.2, 1.0, 1.0, 0.2, 0.0],
  [0.0, 0.1, 0.9, 0.9, 0.1, 0.0],
  [0.0, 0.1, 0.9, 0.9, 0.1, 0.0],
  [0.0, 0.0, 0.2, 0.2, 0.0, 0.0],
];

export const CNN_KERNEL: number[][] = [
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1],
];

export const CNN_DENSE_W: number[][] = [
  [0.65, -0.15],
  [0.42, -0.31],
  [0.58, -0.08],
  [0.47, -0.29],
];

export const CNN_DENSE_B: [number, number] = [0.12, -0.05];

export const MLP_MAX_STEP = 5;
export const CNN_MAX_STEP = 19;

export interface MlpForward {
  z: number;
  a: number;
  y: number;
  loss: number;
}

export interface MlpGradient {
  dw1: number;
  dw2: number;
}

export interface CnnEval {
  conv: number[][];
  pool: number[][];
  flat: number[];
  dense: number[];
  matchScore: number;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function relu(value: number): number {
  return Math.max(0, value);
}

export function softmax(values: number[]): number[] {
  const peak = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - peak));
  const sum = exp.reduce((acc, value) => acc + value, 0);
  return exp.map((value) => value / Math.max(sum, 1e-9));
}

export function computeMlpForward(w: [number, number], b: number, x: [number, number], target: number): MlpForward {
  const z = w[0] * x[0] + w[1] * x[1] + b;
  const a = relu(z);
  const y = a;
  const loss = 0.5 * (y - target) * (y - target);
  return { z, a, y, loss };
}

export function computeMlpGradient(w: [number, number], b: number, x: [number, number], target: number): MlpGradient {
  const forward = computeMlpForward(w, b, x, target);
  const reluPrime = forward.z > 0 ? 1 : 0;
  const dLdy = forward.y - target;
  const dLdz = dLdy * reluPrime;
  return {
    dw1: dLdz * x[0],
    dw2: dLdz * x[1],
  };
}

export function computeMlpLossAtWeights(w1: number, w2: number, b: number, x: [number, number], target: number): number {
  const forward = computeMlpForward([w1, w2], b, x, target);
  return forward.loss;
}

export function dotPatch(source: number[][], kernel: number[][], row: number, col: number): number {
  let sum = 0;
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      sum += (source[row + r]?.[col + c] ?? 0) * (kernel[r]?.[c] ?? 0);
    }
  }
  return sum;
}

export function convValid(source: number[][], kernel: number[][]): number[][] {
  const rows = source.length - 2;
  const cols = source[0].length - 2;
  const out = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      out[row][col] = dotPatch(source, kernel, row, col);
    }
  }
  return out;
}

export function maxPool2x2(source: number[][]): number[][] {
  const rows = Math.floor(source.length / 2);
  const cols = Math.floor(source[0].length / 2);
  const out = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const r = row * 2;
      const c = col * 2;
      out[row][col] = Math.max(source[r][c], source[r][c + 1], source[r + 1][c], source[r + 1][c + 1]);
    }
  }
  return out;
}

export function evaluateCnn(input: number[][], kernel: number[][], scanRow: number, scanCol: number): CnnEval {
  const convRaw = convValid(input, kernel);
  const conv = convRaw.map((row) => row.map((value) => relu(value)));
  const pool = maxPool2x2(conv);
  const flat = pool.flat();
  const logits = [CNN_DENSE_B[0], CNN_DENSE_B[1]];
  for (let i = 0; i < flat.length; i += 1) {
    logits[0] += flat[i] * CNN_DENSE_W[i][0];
    logits[1] += flat[i] * CNN_DENSE_W[i][1];
  }
  const dense = softmax(logits);
  const matchScore = dotPatch(input, kernel, scanRow, scanCol);
  return { conv, pool, flat, dense, matchScore };
}

export function cnnStageFromStep(step: number): CnnStage {
  if (step <= 0) return 'input';
  if (step <= 16) return 'conv';
  if (step === 17) return 'pool';
  if (step === 18) return 'flatten';
  return 'dense';
}

export function cnnScanFromStep(step: number): { row: number; col: number } {
  if (step <= 0) return { row: 0, col: 0 };
  if (step > 16) return { row: 3, col: 3 };
  const index = step - 1;
  return {
    row: Math.floor(index / 4),
    col: index % 4,
  };
}

export function weightColor(weight: number): string {
  if (weight >= 0) {
    const intensity = clamp(weight / 1.5, 0, 1);
    return `rgb(${Math.round(120 + intensity * 135)}, ${Math.round(40 + (1 - intensity) * 20)}, ${Math.round(40 + (1 - intensity) * 20)})`;
  }
  const intensity = clamp(Math.abs(weight) / 1.5, 0, 1);
  return `rgb(${Math.round(25 + (1 - intensity) * 20)}, ${Math.round(60 + (1 - intensity) * 40)}, ${Math.round(140 + intensity * 80)})`;
}
