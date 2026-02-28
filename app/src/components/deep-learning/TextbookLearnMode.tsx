import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';

type LearnMode = 'nn' | 'cnn';
type Playback = 'idle' | 'playing' | 'paused';
type NnPhase =
  | 'input'
  | 'w1'
  | 'relu1'
  | 'w2'
  | 'relu2'
  | 'w3'
  | 'softmax'
  | 'output'
  | 'bpOut'
  | 'bpH2'
  | 'bpH1'
  | 'update';
type CnnStage =
  | 'input'
  | 'conv1'
  | 'act1'
  | 'pool1'
  | 'flatten'
  | 'dense'
  | 'output';

interface NnState {
  phaseIndex: number;
  prevPhaseIndex: number;
  phaseProgress: number;
  trainShift: number;
  logitScale: number;
  trainStep: number;
  cheatHiddenUntilStep: number | null;
  stopAtOutput: boolean;
}

interface CnnState {
  stageIndex: number;
  conv1Scan: number;
  stageProgress: number;
  densePass: 0 | 1;
}

interface LearnState {
  mode: LearnMode;
  playback: Playback;
  tempoMs: number;
  kernelDim: 3 | 5;
  learningRate: number;
  nnWeightDecay: number;
  nnTargetClass: 0 | 1;
  waitMs: number;
  nn: NnState;
  cnn: CnnState;
}

interface Point {
  x: number;
  y: number;
}

interface CellBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

const NN_PHASES: NnPhase[] = ['input', 'w1', 'relu1', 'w2', 'relu2', 'w3', 'softmax', 'output', 'bpOut', 'bpH2', 'bpH1', 'update'];
const CNN_STAGES: CnnStage[] = ['input', 'conv1', 'act1', 'pool1', 'flatten', 'dense', 'output'];

const NN_X = [1.0, 0.5];
const NN_W1 = [
  [0.4, -0.2],
  [0.1, 0.5],
  [0.3, 0.2],
  [-0.4, 0.8],
  [0.7, -0.1],
];
const NN_B1 = [0.1, -0.2, 0.05, 0.1, -0.15];
const NN_W2 = [
  [0.2, -0.1, 0.3, 0.5, -0.2],
  [0.4, 0.1, -0.2, 0.2, 0.3],
  [-0.3, 0.2, 0.1, 0.4, 0.2],
  [0.5, -0.2, 0.3, -0.1, 0.1],
];
const NN_B2 = [0.2, -0.1, 0.05, 0.1];
const NN_W3 = [
  [0.3, -0.2, 0.5, 0.1],
  [-0.1, 0.4, 0.2, 0.3],
];
const NN_B3 = [0.05, -0.03];

function buildSmileyInputGrid(size: number): number[][] {
  const center = (size - 1) / 2;
  const radius = size * 0.42;
  const eyeOffsetX = size * 0.16;
  const eyeY = center - size * 0.12;
  const eyeR = size * 0.05;
  const mouthY = center + size * 0.18;
  const mouthR = size * 0.17;
  const mouthThickness = size * 0.028;

  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, col) => {
      const dx = col - center;
      const dy = row - center;
      const dist = Math.sqrt(dx * dx + dy * dy);

      let value = 12;
      if (dist <= radius) {
        const radial = 1 - dist / radius;
        value = 120 + radial * 130;
      }

      const leftEye = Math.hypot(col - (center - eyeOffsetX), row - eyeY);
      const rightEye = Math.hypot(col - (center + eyeOffsetX), row - eyeY);
      if (leftEye <= eyeR || rightEye <= eyeR) value = 18;

      const arc = Math.hypot(dx, row - mouthY);
      const onSmileArc = Math.abs(arc - mouthR) <= mouthThickness && row >= mouthY - 1 && Math.abs(dx) <= size * 0.18;
      if (onSmileArc) value = 24;

      if (dist > radius + 0.8) value = 8;
      return Math.round(clamp(value, 0, 255));
    })
  );
}

const INPUT_GRID = buildSmileyInputGrid(28);

const NN_ANIM_MS = 1100;
const CNN_ANIM_MS = 90;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function relu(value: number): number {
  return Math.max(0, value);
}

function softmax(values: number[]): number[] {
  const peak = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - peak));
  const sum = exp.reduce((acc, value) => acc + value, 0);
  return exp.map((value) => value / Math.max(sum, 1e-9));
}

function nnTargetVector(targetClass: 0 | 1): [number, number] {
  return targetClass === 0 ? [1, 0] : [0, 1];
}

function crossEntropyLoss(probs: number[], target: [number, number]): number {
  return -target.reduce((acc, value, idx) => acc + value * Math.log(Math.max(probs[idx] ?? 1e-9, 1e-9)), 0);
}

function buildVerticalKernel(size: number): number[][] {
  const mid = Math.floor(size / 2);
  return Array.from({ length: size }, () =>
    Array.from({ length: size }, (_, col) => {
      if (col === 0) return -1;
      if (col === mid) return 0;
      if (col === size - 1) return 1;
      return 0;
    })
  );
}

function cnnDims(kernelDim: 3 | 5): {
  input: number;
  kernel1: number;
  conv1: number;
  pool1: number;
} {
  const input = INPUT_GRID.length;
  const kernel1 = kernelDim;
  const conv1 = input - kernel1 + 1;
  const pool1 = Math.floor((conv1 - 2) / 2) + 1;
  return { input, kernel1, conv1, pool1 };
}

function dotPatch(source: number[][], kernel: number[][], row: number, col: number): number {
  let sum = 0;
  for (let r = 0; r < kernel.length; r += 1) {
    for (let c = 0; c < kernel[0].length; c += 1) {
      sum += (source[row + r]?.[col + c] ?? 0) * (kernel[r]?.[c] ?? 0);
    }
  }
  return Number(sum.toFixed(3));
}

function convValid(source: number[][], kernel: number[][]): number[][] {
  const outRows = source.length - kernel.length + 1;
  const outCols = source[0].length - kernel[0].length + 1;
  const out = Array.from({ length: outRows }, () => Array.from({ length: outCols }, () => 0));
  for (let row = 0; row < outRows; row += 1) {
    for (let col = 0; col < outCols; col += 1) {
      out[row][col] = dotPatch(source, kernel, row, col);
    }
  }
  return out;
}

function maxPool(source: number[][], stride: number): number[][] {
  const outRows = Math.floor((source.length - 2) / stride) + 1;
  const outCols = Math.floor((source[0].length - 2) / stride) + 1;
  const out = Array.from({ length: outRows }, () => Array.from({ length: outCols }, () => 0));
  for (let row = 0; row < outRows; row += 1) {
    for (let col = 0; col < outCols; col += 1) {
      const r = row * stride;
      const c = col * stride;
      out[row][col] = Math.max(source[r][c], source[r][c + 1], source[r + 1][c], source[r + 1][c + 1]);
    }
  }
  return out.map((line) => line.map((value) => Number(value.toFixed(3))));
}

function addVec(vec: number[], bias: number[]): number[] {
  return vec.map((value, index) => value + (bias[index] ?? 0));
}

function matVec(matrix: number[][], vec: number[]): number[] {
  return matrix.map((row) => row.reduce((acc, value, idx) => acc + value * (vec[idx] ?? 0), 0));
}

function fmt(values: number[]): string {
  return values.map((value) => value.toFixed(2)).join(', ');
}

function computeNnForward(trainShift: number, logitScale: number): {
  z1: number[];
  a1: number[];
  z2: number[];
  a2: number[];
  z3Base: number[];
  z3: number[];
  probs: number[];
} {
  const z1 = addVec(matVec(NN_W1, NN_X), NN_B1);
  const a1 = z1.map(relu);
  const z2 = addVec(matVec(NN_W2, a1), NN_B2);
  const a2 = z2.map(relu);
  const z3Base = addVec(matVec(NN_W3, a2), NN_B3);
  const z3 = [z3Base[0] * logitScale + trainShift, z3Base[1] * logitScale - trainShift];
  const probs = softmax(z3);
  return { z1, a1, z2, a2, z3Base, z3, probs };
}

function applyBackpropStep(
  trainShift: number,
  logitScale: number,
  learningRate: number,
  targetClass: 0 | 1,
  weightDecay: number
): {
  nextShift: number;
  nextLogitScale: number;
  gradShift: number;
  gradScale: number;
  lossBefore: number;
  lossAfter: number;
  before: ReturnType<typeof computeNnForward>;
  after: ReturnType<typeof computeNnForward>;
} {
  const target = nnTargetVector(targetClass);
  const before = computeNnForward(trainShift, logitScale);
  const d0 = before.probs[0] - target[0];
  const d1 = before.probs[1] - target[1];
  const gradShift = d0 - d1;
  const gradScale = d0 * before.z3Base[0] + d1 * before.z3Base[1];
  const decayGrad = weightDecay * (logitScale - 1);
  const nextShift = clamp(trainShift - learningRate * gradShift, -4, 4);
  const nextLogitScale = clamp(logitScale - learningRate * (gradScale + decayGrad), 0.35, 3.5);
  const after = computeNnForward(nextShift, nextLogitScale);
  const lossBefore = crossEntropyLoss(before.probs, target);
  const lossAfter = crossEntropyLoss(after.probs, target);
  return { nextShift, nextLogitScale, gradShift, gradScale, lossBefore, lossAfter, before, after };
}

function wrongStartForTarget(targetClass: 0 | 1): { trainShift: number; logitScale: number } {
  const neutral = computeNnForward(0, 1);
  const baseDelta = neutral.z3Base[0] - neutral.z3Base[1];
  const desiredDelta = targetClass === 0 ? -0.06 : 0.06;
  const trainShift = clamp((desiredDelta - baseDelta) * 0.5, -4, 4);
  return { trainShift, logitScale: 1 };
}

function buildNnInit(targetClass: 0 | 1): NnState {
  const wrong = wrongStartForTarget(targetClass);
  return {
    phaseIndex: 0,
    prevPhaseIndex: 0,
    phaseProgress: 1,
    trainShift: wrong.trainShift,
    logitScale: wrong.logitScale,
    trainStep: 0,
    cheatHiddenUntilStep: null,
    stopAtOutput: false,
  };
}

function hideNnCheat(nn: NnState): NnState {
  const targetStep = nn.trainStep + 1;
  return {
    ...nn,
    cheatHiddenUntilStep: Math.max(nn.cheatHiddenUntilStep ?? targetStep, targetStep),
  };
}

function initialState(): LearnState {
  const nnTargetClass: 0 | 1 = 0;
  return {
    mode: 'nn',
    playback: 'idle',
    tempoMs: 420,
    kernelDim: 3,
    learningRate: 0.08,
    nnWeightDecay: 0.02,
    nnTargetClass,
    waitMs: 0,
    nn: buildNnInit(nnTargetClass),
    cnn: {
      stageIndex: 0,
      conv1Scan: 0,
      stageProgress: 1,
      densePass: 0,
    },
  };
}

function advanceNn(state: LearnState): LearnState {
  const nextIndex = Math.min(NN_PHASES.length - 1, state.nn.phaseIndex + 1);
  return {
    ...state,
    nn: {
      ...state.nn,
      phaseIndex: nextIndex,
      prevPhaseIndex: state.nn.phaseIndex,
      phaseProgress: 0,
    },
  };
}

function advanceCnn(state: LearnState): LearnState {
  const dims = cnnDims(state.kernelDim);
  const stage = CNN_STAGES[state.cnn.stageIndex];
  const fastHopFor = (width: number): number => {
    if (state.playback !== 'playing') return 1;
    if (width >= 24) return 3;
    if (width >= 16) return 2;
    return 1;
  };
  if (stage === 'conv1') {
    const maxIndex = dims.conv1 * dims.conv1 - 1;
    if (state.cnn.conv1Scan < maxIndex) {
      const hop = fastHopFor(dims.conv1);
      return {
        ...state,
        cnn: {
          ...state.cnn,
          conv1Scan: Math.min(maxIndex, state.cnn.conv1Scan + hop),
          stageProgress: 0,
        },
      };
    }
  }

  if (stage === 'dense') {
    if (state.cnn.densePass === 0) {
      return {
        ...state,
        cnn: {
          ...state.cnn,
          densePass: 1,
          stageProgress: 0,
        },
      };
    }
  }

  const nextStage = Math.min(CNN_STAGES.length - 1, state.cnn.stageIndex + 1);
  return {
    ...state,
    cnn: {
      ...state.cnn,
      stageIndex: nextStage,
      stageProgress: 0,
      densePass: nextStage === CNN_STAGES.indexOf('dense') ? state.cnn.densePass : 0,
    },
  };
}

function skipCnnLayer(state: LearnState): LearnState {
  const dims = cnnDims(state.kernelDim);
  const stage = CNN_STAGES[state.cnn.stageIndex];
  const to = (nextStage: CnnStage, patch?: Partial<CnnState>): LearnState => ({
    ...state,
    cnn: {
      ...state.cnn,
      stageIndex: CNN_STAGES.indexOf(nextStage),
      stageProgress: 1,
      ...patch,
    },
  });

  if (stage === 'input') return to('conv1', { densePass: 0 });
  if (stage === 'conv1') return to('act1', { conv1Scan: Math.max(0, dims.conv1 * dims.conv1 - 1), densePass: 0 });
  if (stage === 'act1') return to('pool1', { densePass: 0 });
  if (stage === 'pool1') return to('flatten', { densePass: 0 });
  if (stage === 'flatten') return to('dense', { densePass: 0 });
  if (stage === 'dense') {
    if (state.cnn.densePass === 0) return to('dense', { densePass: 1 });
    return to('output', { densePass: 0 });
  }

  return state;
}

function tickState(state: LearnState, dt: number): LearnState {
  let next = state;
  const nnPhase = NN_PHASES[next.nn.phaseIndex];
  const nnPhaseDuration =
    nnPhase === 'bpOut' || nnPhase === 'bpH2' || nnPhase === 'bpH1'
      ? NN_ANIM_MS * 1.9
      : nnPhase === 'update'
        ? NN_ANIM_MS * 1.55
        : NN_ANIM_MS;

  if (next.nn.phaseProgress < 1) {
    next = {
      ...next,
      nn: {
        ...next.nn,
        phaseProgress: clamp(next.nn.phaseProgress + dt / nnPhaseDuration, 0, 1),
      },
    };
  }

  if (
    next.mode === 'nn' &&
    next.nn.cheatHiddenUntilStep !== null &&
    NN_PHASES[next.nn.phaseIndex] === 'output' &&
    next.nn.phaseProgress >= 1 &&
    next.nn.trainStep >= next.nn.cheatHiddenUntilStep
  ) {
    next = {
      ...next,
      nn: {
        ...next.nn,
        cheatHiddenUntilStep: null,
      },
    };
  }

  if (next.cnn.stageProgress < 1) {
    const currentCnnStage = CNN_STAGES[next.cnn.stageIndex];
    const convDuration = clamp(next.tempoMs * 0.34, 32, 260);
    const stageDuration =
      currentCnnStage === 'conv1'
        ? convDuration
        : currentCnnStage === 'dense'
          ? CNN_ANIM_MS * 17
          : currentCnnStage === 'flatten'
            ? CNN_ANIM_MS * 14.5
            : currentCnnStage === 'pool1'
              ? CNN_ANIM_MS * 5.2
              : currentCnnStage === 'act1'
                ? CNN_ANIM_MS * 4.4
                : CNN_ANIM_MS;
    next = {
      ...next,
      cnn: {
        ...next.cnn,
        stageProgress: clamp(next.cnn.stageProgress + dt / stageDuration, 0, 1),
      },
    };
  }

  if (next.mode === 'nn' && NN_PHASES[next.nn.phaseIndex] === 'update' && next.nn.phaseProgress >= 1) {
    if (next.playback !== 'playing') {
      return next;
    }
    const trained = applyBackpropStep(
      next.nn.trainShift,
      next.nn.logitScale,
      next.learningRate,
      next.nnTargetClass,
      next.nnWeightDecay
    );
    next = {
      ...next,
      playback: next.playback,
      waitMs: next.tempoMs,
      nn: {
        ...next.nn,
        trainShift: trained.nextShift,
        logitScale: trained.nextLogitScale,
        trainStep: next.nn.trainStep + 1,
        prevPhaseIndex: next.nn.phaseIndex,
        phaseIndex: NN_PHASES.indexOf('input'),
        phaseProgress: 0,
        stopAtOutput: true,
      },
    };
  }

  if (
    next.mode === 'nn' &&
    next.playback === 'playing' &&
    next.nn.stopAtOutput &&
    NN_PHASES[next.nn.phaseIndex] === 'output' &&
    next.nn.phaseProgress >= 1
  ) {
    return {
      ...next,
      playback: 'paused',
      waitMs: 0,
    };
  }

  if (next.playback !== 'playing') return next;

  if (next.waitMs > 0) {
    return {
      ...next,
      waitMs: Math.max(0, next.waitMs - dt),
    };
  }

  if (next.mode === 'nn') {
    if (next.nn.phaseProgress < 1) return next;
    return {
      ...advanceNn(next),
      waitMs: next.tempoMs * 1.35,
    };
  }

  if (next.cnn.stageProgress < 1) return next;
  const atEnd = next.cnn.stageIndex >= CNN_STAGES.length - 1 && CNN_STAGES[next.cnn.stageIndex] !== 'conv1';
  if (atEnd) {
    return {
      ...next,
      playback: 'paused',
    };
  }

  const nextStage = CNN_STAGES[next.cnn.stageIndex];
  const dwellMs =
    nextStage === 'conv1'
      ? clamp(next.tempoMs * 0.28, 36, 190)
      : nextStage === 'pool1'
        ? 420
        : nextStage === 'flatten'
          ? 1360
        : nextStage === 'dense'
          ? 1640
          : nextStage === 'act1'
            ? 520
        : 480;

  return {
    ...advanceCnn(next),
    waitMs: dwellMs,
  };
}

function pointOnPath(points: Point[], t: number): Point {
  if (points.length === 0) return { x: 0, y: 0 };
  if (points.length === 1) return points[0];
  const clamped = clamp(t, 0, 1);
  const scaled = clamped * (points.length - 1);
  const seg = Math.min(points.length - 2, Math.floor(scaled));
  const localT = scaled - seg;
  const start = points[seg];
  const end = points[seg + 1];
  return {
    x: start.x + (end.x - start.x) * localT,
    y: start.y + (end.y - start.y) * localT,
  };
}

function lineProgress(phaseIndex: number, phaseProgress: number, groupPhase: 'w1' | 'w2' | 'w3'): number {
  const current = NN_PHASES[phaseIndex];
  const groupIdx = groupPhase === 'w1' ? 1 : groupPhase === 'w2' ? 3 : 5;
  const currentIdx = NN_PHASES.indexOf(current);
  if (current === 'w1' || current === 'w2' || current === 'w3') {
    if (currentIdx > groupIdx) return 1;
    if (currentIdx === groupIdx) return phaseProgress;
    return 0;
  }
  if (current === 'bpOut') return groupPhase === 'w3' ? clamp(1 - phaseProgress, 0, 1) : 1;
  if (current === 'bpH2') return groupPhase === 'w2' ? clamp(1 - phaseProgress, 0, 1) : 1;
  if (current === 'bpH1') return groupPhase === 'w1' ? clamp(1 - phaseProgress, 0, 1) : 1;
  if (current === 'update') return 1;
  if (currentIdx > groupIdx) return 1;
  return 0;
}

function mapScan(index: number, width: number): { row: number; col: number } {
  const safeWidth = Math.max(width, 1);
  const row = Math.floor(index / safeWidth);
  const offset = index % safeWidth;
  return {
    row,
    col: row % 2 === 0 ? offset : safeWidth - 1 - offset,
  };
}

function serpentineOrderIndex(row: number, col: number, width: number): number {
  const safeWidth = Math.max(width, 1);
  return row * safeWidth + (row % 2 === 0 ? col : safeWidth - 1 - col);
}

function matrixRange(matrix: number[][]): { min: number; max: number } {
  const flat = matrix.flat();
  return {
    min: Math.min(...flat),
    max: Math.max(...flat),
  };
}

function heatColor(value: number, min: number, max: number): string {
  const span = Math.max(max - min, 1e-6);
  const t = clamp((value - min) / span, 0, 1);
  const hue = 222 - t * 210;
  const sat = 76;
  const light = 30 + t * 22;
  return `hsl(${hue.toFixed(1)} ${sat}% ${light.toFixed(1)}%)`;
}

function heatTextColor(value: number, min: number, max: number): string {
  const span = Math.max(max - min, 1e-6);
  const t = clamp((value - min) / span, 0, 1);
  return t > 0.54 ? 'var(--bg, #ffffff)' : 'var(--text, #0f172a)';
}

function formatCellValue(value: number): string {
  if (Number.isInteger(value)) return value.toString();
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function cellBoxesFromGrid(grid: HTMLElement, rootRect: DOMRect): CellBox[] {
  return Array.from(grid.children).map((child) => {
    const rect = (child as HTMLElement).getBoundingClientRect();
    return {
      left: rect.left - rootRect.left,
      top: rect.top - rootRect.top,
      width: rect.width,
      height: rect.height,
    };
  });
}

function gridStyle(cols: number): CSSProperties {
  return {
    ['--cnn-cell-size' as string]: '5px',
    gridTemplateColumns: `repeat(${Math.max(cols, 1)}, var(--cnn-cell-size, 6px))`,
  };
}

interface NnCanvasNode {
  id: string;
  layer: 0 | 1 | 2 | 3;
  index: number;
  x: number;
  y: number;
  hue: number;
}

interface NnCanvasEdge {
  sourceId: string;
  targetId: string;
  group: 'w1' | 'w2' | 'w3';
  weight: number;
}

interface NnPhaseMeta {
  label: string;
  english: string;
  math: string;
  example: string;
  forward: boolean;
}

function buildNnPhaseMeta(
  phase: NnPhase,
  nnForward: ReturnType<typeof computeNnForward>,
  trainShift: number,
  logitScale: number,
  trainStep: number,
  learningRate: number,
  targetClass: 0 | 1,
  weightDecay: number
): NnPhaseMeta {
  const phaseLabel =
    phase === 'input'
      ? 'Making a Guess (Forward Pass)'
      : phase === 'w1' || phase === 'w2' || phase === 'w3'
        ? 'Linear Transform + Bias'
        : phase === 'relu1' || phase === 'relu2'
          ? 'Activation (ReLU)'
          : phase === 'softmax'
            ? 'Probability Mapping'
            : phase === 'output'
              ? 'Final Prediction'
              : phase === 'bpOut'
                ? 'Backpropagation: Output Delta'
                : phase === 'bpH2'
                  ? 'Backpropagation: Hidden Layer 2'
                  : phase === 'bpH1'
                    ? 'Backpropagation: Hidden Layer 1'
                    : 'Gradient Descent Weight Update';

  const english =
    phase === 'input'
      ? 'Input features enter the network and get routed to hidden neurons.'
      : phase === 'w1'
        ? 'Hidden layer 1 computes weighted sums from the raw inputs.'
        : phase === 'relu1'
          ? 'ReLU clips negative values and keeps useful positive evidence.'
          : phase === 'w2'
            ? 'Hidden layer 2 remixes first-layer features into deeper patterns.'
            : phase === 'relu2'
              ? 'Second ReLU preserves strong features and removes weak negatives.'
              : phase === 'w3'
                ? 'Output logits are computed from hidden layer 2 activations.'
                : phase === 'softmax'
                  ? 'Softmax converts logits into normalized class probabilities.'
                  : phase === 'output'
                    ? 'The model selects the class with highest probability.'
                    : phase === 'bpOut'
                      ? 'Compute output-layer error: how far probabilities are from the target label.'
                      : phase === 'bpH2'
                        ? 'Backpropagate that error into hidden layer 2 through W3.'
                        : phase === 'bpH1'
                          ? 'Propagate hidden error further into hidden layer 1 through W2.'
                          : 'Apply gradient descent to update weights and lower the next loss.';

  const math =
    phase === 'input'
      ? 'x = [x1, x2]'
      : phase === 'w1'
        ? 'z1 = W1·x + b1'
        : phase === 'relu1'
          ? 'a1 = ReLU(z1)'
          : phase === 'w2'
            ? 'z2 = W2·a1 + b2'
            : phase === 'relu2'
              ? 'a2 = ReLU(z2)'
              : phase === 'w3'
                ? 'z3 = W3·a2 + b3'
                : phase === 'softmax'
                  ? 'p = softmax(z3)'
                  : phase === 'output'
                    ? `y = argmax(p), p=[${(nnForward.probs[0] * 100).toFixed(1)}%, ${(nnForward.probs[1] * 100).toFixed(1)}%]`
                    : phase === 'bpOut'
                      ? 'δ3 = p - y'
                      : phase === 'bpH2'
                        ? "δ2 = (W3^T δ3) ⊙ ReLU'(z2)"
                        : phase === 'bpH1'
                          ? "δ1 = (W2^T δ2) ⊙ ReLU'(z1)"
                          : `θ ← θ - η·(∇L + λ·reg), η=${learningRate.toFixed(2)}, λ=${weightDecay.toFixed(2)}`;

  const buildExample = (layer: 1 | 2 | 3): string => {
    const source = layer === 1 ? NN_X : layer === 2 ? nnForward.a1 : nnForward.a2;
    const weights = layer === 1 ? NN_W1[0] : layer === 2 ? NN_W2[0] : NN_W3[0];
    const bias = layer === 1 ? NN_B1[0] : layer === 2 ? NN_B2[0] : NN_B3[0];
    const sum = source.reduce((acc, value, idx) => acc + value * (weights[idx] ?? 0), 0);
    const shift = layer === 3 ? trainShift : 0;
    const scale = layer === 3 ? logitScale : 1;
    const z = (sum + bias) * scale + shift;
    const a = layer === 3 ? nnForward.probs[0] : relu(z);

    const terms = source
      .slice(0, 4)
      .map((value, idx) => `In${idx + 1}(${value.toFixed(2)}) × W(${(weights[idx] ?? 0).toFixed(2)}) = ${(value * (weights[idx] ?? 0)).toFixed(2)}`)
      .join('\n');
    const shiftLine = layer === 3 ? `\nlogit scale = ${scale.toFixed(2)} | shift = ${shift.toFixed(2)}` : '';
    return `${terms}\n\nsum + b = ${sum.toFixed(2)} + ${bias.toFixed(2)} = ${(sum + bias).toFixed(2)}${shiftLine}\noutput = ${a.toFixed(3)}`;
  };

  const example =
    phase === 'input'
      ? `x = [${fmt(NN_X)}]\nSample enters two input neurons as normalized features.`
      : phase === 'w1'
        ? `Neuron 1 (L1)\n${buildExample(1)}`
        : phase === 'relu1'
          ? `a1 = ReLU(z1)\n[${fmt(nnForward.a1)}]`
          : phase === 'w2'
            ? `Neuron 1 (L2)\n${buildExample(2)}`
            : phase === 'relu2'
              ? `a2 = ReLU(z2)\n[${fmt(nnForward.a2)}]`
              : phase === 'w3'
                ? `Neuron 1 (Output logit)\n${buildExample(3)}`
                : phase === 'softmax'
                  ? `softmax(z3)\nClass0 ${(nnForward.probs[0] * 100).toFixed(1)}%\nClass1 ${(nnForward.probs[1] * 100).toFixed(1)}%`
                  : phase === 'output'
                    ? (() => {
                        const target = nnTargetVector(targetClass);
                        const loss = crossEntropyLoss(nnForward.probs, target);
                        return `Prediction: class ${nnForward.probs[0] >= nnForward.probs[1] ? 0 : 1}\nTarget class ${targetClass}\nLoss ${loss.toFixed(4)}\nStep ${trainStep}`;
                      })()
                    : phase === 'bpOut'
                      ? (() => {
                          const target = nnTargetVector(targetClass);
                          const d3 = [nnForward.probs[0] - target[0], nnForward.probs[1] - target[1]];
                          return `target = [${target[0]}, ${target[1]}]\nδ3 = p - y = [${d3.map((v) => v.toFixed(3)).join(', ')}]`;
                        })()
                      : phase === 'bpH2'
                        ? 'Hidden2 receives error through W3^T and ReLU gate.\nOnly active neurons pass gradient.'
                        : phase === 'bpH1'
                          ? 'Hidden1 receives error through W2^T.\nInput-connected weights get credit/blame.'
                          : `Update with η=${learningRate.toFixed(2)}, λ=${weightDecay.toFixed(2)}\nTrainable params: [shift, logitScale].`;

  return {
    label: phaseLabel,
    english,
    math,
    example,
    forward:
      phase === 'input' ||
      phase === 'w1' ||
      phase === 'relu1' ||
      phase === 'w2' ||
      phase === 'relu2' ||
      phase === 'w3' ||
      phase === 'softmax' ||
      phase === 'output',
  };
}

function computeNnLayout(width: number, height: number): { nodes: NnCanvasNode[]; edges: NnCanvasEdge[] } {
  const layerSizes = [2, 5, 4, 2] as const;
  const isNarrow = width <= 720;
  const padY = Math.max(26, height * 0.1);
  const rawUsableW = Math.max(240, width - 120);
  const rawUsableH = Math.max(150, height - padY * 2);
  const usableW = Math.max(isNarrow ? 236 : 220, rawUsableW * (isNarrow ? 0.72 : 0.5));
  const usableH = Math.max(isNarrow ? 132 : 120, rawUsableH * (isNarrow ? 0.58 : 0.5));
  const maxLayerSize = Math.max(...layerSizes);
  const dx = usableW / (layerSizes.length - 1);
  const startX = (width - usableW) * 0.5;
  const nodes: NnCanvasNode[] = [];

  let hueIndex = 0;
  const total = layerSizes.reduce((acc, value) => acc + value, 0);
  for (let layer = 0; layer < layerSizes.length; layer += 1) {
    const count = layerSizes[layer];
    const layerSpan = (count - 1) * (usableH / Math.max(maxLayerSize - 1, 1));
    const startY = (height - layerSpan) / 2;
    for (let index = 0; index < count; index += 1) {
      nodes.push({
        id: `${layer}-${index}`,
        layer: layer as 0 | 1 | 2 | 3,
        index,
        x: startX + layer * dx,
        y: startY + index * (layerSpan / Math.max(count - 1, 1)),
        hue: (hueIndex++ * (360 / total)) % 360,
      });
    }
  }

  const byLayer = (layer: number) => nodes.filter((node) => node.layer === layer);
  const edges: NnCanvasEdge[] = [];
  byLayer(0).forEach((source) => {
    byLayer(1).forEach((target) => {
      edges.push({ sourceId: source.id, targetId: target.id, group: 'w1', weight: NN_W1[target.index]?.[source.index] ?? 0 });
    });
  });
  byLayer(1).forEach((source) => {
    byLayer(2).forEach((target) => {
      edges.push({ sourceId: source.id, targetId: target.id, group: 'w2', weight: NN_W2[target.index]?.[source.index] ?? 0 });
    });
  });
  byLayer(2).forEach((source) => {
    byLayer(3).forEach((target) => {
      edges.push({ sourceId: source.id, targetId: target.id, group: 'w3', weight: NN_W3[target.index]?.[source.index] ?? 0 });
    });
  });
  return { nodes, edges };
}

function nnPhasePath(phase: NnPhase, nodes: NnCanvasNode[]): Point[] {
  const input = nodes.filter((node) => node.layer === 0).sort((a, b) => a.index - b.index);
  const h1 = nodes.filter((node) => node.layer === 1).sort((a, b) => a.index - b.index);
  const h2 = nodes.filter((node) => node.layer === 2).sort((a, b) => a.index - b.index);
  const out = nodes.filter((node) => node.layer === 3).sort((a, b) => a.index - b.index);

  if (phase === 'input') return [input[0], input[1]];
  if (phase === 'w1') return [input[1], ...h1];
  if (phase === 'relu1') return [...h1].reverse();
  if (phase === 'w2') return [h1[0], ...h2];
  if (phase === 'relu2') return [...h2].reverse();
  if (phase === 'w3') return [h2[0], out[0], out[1]];
  if (phase === 'softmax' || phase === 'output') return [out[1], out[0], out[1]];
  if (phase === 'bpOut') return [out[1], out[0], ...h2.slice().reverse()];
  if (phase === 'bpH2') return [h2[0], ...h1.slice().reverse()];
  if (phase === 'bpH1') return [h1[0], input[1], input[0]];
  return [input[0], ...h1, ...h2, out[0], out[1]];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function GodNnCanvasStage({
  phase,
  phaseProgress,
  nnForward,
  trainShift,
  logitScale,
  trainStep,
  learningRate,
  targetClass,
  weightDecay,
  showCheat,
  onAnyClick,
}: {
  phase: NnPhase;
  phaseProgress: number;
  nnForward: ReturnType<typeof computeNnForward>;
  trainShift: number;
  logitScale: number;
  trainStep: number;
  learningRate: number;
  targetClass: 0 | 1;
  weightDecay: number;
  showCheat: boolean;
  onAnyClick: () => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ width: 760, height: 360 });
  const [pointer, setPointer] = useState<{ x: number; y: number }>({ x: 380, y: 180 });
  const [hovered, setHovered] = useState<NnCanvasNode | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 10, y: 10 });
  const panRef = useRef({ x: 0, y: 0 });
  const layout = useMemo(() => computeNnLayout(size.width, size.height), [size.height, size.width]);
  const nodeMap = useMemo(() => new Map(layout.nodes.map((node) => [node.id, node])), [layout.nodes]);
  const phaseMeta = useMemo(
    () => buildNnPhaseMeta(phase, nnForward, trainShift, logitScale, trainStep, learningRate, targetClass, weightDecay),
    [learningRate, logitScale, nnForward, phase, targetClass, trainShift, trainStep, weightDecay]
  );
  const packet = useMemo(() => pointOnPath(nnPhasePath(phase, layout.nodes), phaseProgress), [layout.nodes, phase, phaseProgress]);
  const topFloatWidth = Math.min(430, Math.max(260, size.width - 22));
  const bottomFloatWidth = Math.min(430, Math.max(260, size.width - 22));
  const reserveLeft = size.width > 760 && showCheat ? 226 : 0;
  const topMinX = Math.max(10 + topFloatWidth / 2, reserveLeft + topFloatWidth / 2 + 8);
  const topMaxX = Math.max(topMinX, size.width - 10 - topFloatWidth / 2);
  const bottomMinX = 10 + bottomFloatWidth / 2;
  const bottomMaxX = Math.max(bottomMinX, size.width - 10 - bottomFloatWidth / 2);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const resize = () => {
      const rect = element.getBoundingClientRect();
      setSize({
        width: Math.max(300, Math.floor(rect.width)),
        height: Math.max(300, Math.floor(rect.height)),
      });
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let raf = 0;
    const render = (time: number) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        raf = window.requestAnimationFrame(render);
        return;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        raf = window.requestAnimationFrame(render);
        return;
      }
      if (canvas.width !== size.width || canvas.height !== size.height) {
        canvas.width = size.width;
        canvas.height = size.height;
      }

      const dark = document.documentElement.classList.contains('dark');
      const centerX = size.width * 0.5;
      const centerY = size.height * 0.5;
      const targetPanX = (centerX - pointer.x) * 0.025;
      const targetPanY = (centerY - pointer.y) * 0.025;
      panRef.current.x = lerp(panRef.current.x, targetPanX, 0.09);
      panRef.current.y = lerp(panRef.current.y, targetPanY, 0.09);

      const panX = panRef.current.x;
      const panY = panRef.current.y;
      const phaseIndex = NN_PHASES.indexOf(phase);
      const forwardSignalGroup = phase === 'w1' || phase === 'w2' || phase === 'w3' ? phase : null;
      const backpropSignalGroup = phase === 'bpOut' ? 'w3' : phase === 'bpH2' ? 'w2' : phase === 'bpH1' ? 'w1' : null;
      const isBackpropPhase = backpropSignalGroup !== null || phase === 'update';

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(panX, panY);

      layout.edges.forEach((edge) => {
        const source = nodeMap.get(edge.sourceId);
        const target = nodeMap.get(edge.targetId);
        if (!source || !target) return;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        const baseAlpha = dark ? 0.08 : 0.1;
        ctx.strokeStyle = `rgba(120, 136, 170, ${baseAlpha})`;
        ctx.lineWidth = 1 + Math.abs(edge.weight) * 1.3;
        ctx.stroke();
      });

      layout.edges.forEach((edge) => {
        const source = nodeMap.get(edge.sourceId);
        const target = nodeMap.get(edge.targetId);
        if (!source || !target) return;
        const p = lineProgress(phaseIndex, phaseProgress, edge.group);
        if (p <= 0) return;
        const x2 = lerp(source.x, target.x, p);
        const y2 = lerp(source.y, target.y, p);
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(x2, y2);
        if (isBackpropPhase) {
          ctx.setLineDash([5, 3]);
          ctx.lineDashOffset = -((time / 24) % 12);
        } else {
          ctx.setLineDash([]);
        }
        ctx.strokeStyle = `hsla(${target.hue.toFixed(1)}, 88%, 52%, ${isBackpropPhase ? 0.72 : 0.85})`;
        ctx.lineWidth = isBackpropPhase ? 1.9 : 2.2;
        ctx.stroke();
        ctx.setLineDash([]);
      });

      if (forwardSignalGroup || backpropSignalGroup) {
        const signalGroup = (forwardSignalGroup ?? backpropSignalGroup) as 'w1' | 'w2' | 'w3';
        const reverse = Boolean(backpropSignalGroup);
        layout.edges.forEach((edge) => {
          if (edge.group !== signalGroup) return;
          const source = nodeMap.get(edge.sourceId);
          const target = nodeMap.get(edge.targetId);
          if (!source || !target) return;
          const t = clamp(phaseProgress, 0, 1);
          const sx = reverse ? target.x : source.x;
          const sy = reverse ? target.y : source.y;
          const tx = reverse ? source.x : target.x;
          const ty = reverse ? source.y : target.y;
          const x = lerp(sx, tx, t);
          const y = lerp(sy, ty, t);
          const tailT = Math.max(0, t - 0.15);
          const tailX = lerp(sx, tx, tailT);
          const tailY = lerp(sy, ty, tailT);
          const grad = ctx.createLinearGradient(tailX, tailY, x, y);
          grad.addColorStop(0, 'rgba(0,0,0,0)');
          grad.addColorStop(1, `hsla(${source.hue.toFixed(1)}, 92%, 58%, 0.9)`);
          ctx.beginPath();
          ctx.moveTo(tailX, tailY);
          ctx.lineTo(x, y);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 3.2;
          ctx.lineCap = 'round';
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x, y, 3.4, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.shadowColor = `hsla(${source.hue.toFixed(1)}, 92%, 58%, 0.95)`;
          ctx.shadowBlur = 10;
          ctx.fill();
          ctx.shadowBlur = 0;
        });
      } else {
        ctx.beginPath();
        ctx.arc(packet.x, packet.y, 5.3, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(14, 165, 233, 0.95)';
        ctx.shadowBlur = 14;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      layout.nodes.forEach((node) => {
        const r = node.layer === 0 || node.layer === 3 ? 14 : 12;
        const isHover = hovered?.id === node.id;
        const gradient = ctx.createRadialGradient(node.x - 2, node.y - 2, 0, node.x, node.y, r);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.4, `hsla(${node.hue.toFixed(1)}, 90%, 64%, 1)`);
        gradient.addColorStop(1, `hsla(${node.hue.toFixed(1)}, 88%, ${dark ? 34 : 42}%, 0.9)`);
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
        if (isHover) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, r + 3, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${node.hue.toFixed(1)}, 95%, 56%, 0.28)`;
          ctx.fill();
        }
        ctx.strokeStyle = dark ? 'rgba(226,232,240,0.35)' : 'rgba(15,23,42,0.2)';
        ctx.lineWidth = isHover ? 2.4 : 1.7;
        ctx.stroke();
      });

      const layerNames = ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer'] as const;
      for (let layer = 0; layer < 4; layer += 1) {
        const layerNodes = layout.nodes.filter((node) => node.layer === layer);
        if (!layerNodes.length) continue;
        const x = layerNodes.reduce((acc, node) => acc + node.x, 0) / layerNodes.length;
        const topY = Math.min(...layerNodes.map((node) => node.y));
        ctx.fillStyle = dark ? 'rgba(148, 163, 184, 0.86)' : 'rgba(71, 85, 105, 0.78)';
        ctx.font = '600 11px Inter, Roboto, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(layerNames[layer], x, topY - 18);
      }

      ctx.restore();
      raf = window.requestAnimationFrame(render);
    };
    raf = window.requestAnimationFrame(render);
    return () => window.cancelAnimationFrame(raf);
  }, [layout.edges, layout.nodes, nodeMap, packet.x, packet.y, phase, phaseProgress, pointer.x, pointer.y, size.height, size.width, hovered?.id]);

  const tooltipData = useMemo(() => {
    if (!hovered) return null;
    const layerLabel = hovered.layer === 0 ? 'Input' : hovered.layer === 1 ? 'Hidden 1' : hovered.layer === 2 ? 'Hidden 2' : 'Output';
    if (hovered.layer === 0) {
      return {
        title: `${layerLabel} - Neuron ${hovered.index + 1}`,
        description: 'Input neurons pass raw features into the network.',
        rows: [`x${hovered.index + 1} = ${(NN_X[hovered.index] ?? 0).toFixed(2)}`],
        output: (NN_X[hovered.index] ?? 0).toFixed(2),
      };
    }

    const prevValues = hovered.layer === 1 ? NN_X : hovered.layer === 2 ? nnForward.a1 : nnForward.a2;
    const weights = hovered.layer === 1 ? NN_W1[hovered.index] : hovered.layer === 2 ? NN_W2[hovered.index] : NN_W3[hovered.index];
    const bias = hovered.layer === 1 ? NN_B1[hovered.index] : hovered.layer === 2 ? NN_B2[hovered.index] : NN_B3[hovered.index];
    const products = prevValues.map((value, index) => ({ value, weight: weights?.[index] ?? 0, product: value * (weights?.[index] ?? 0) }));
    const sum = products.reduce((acc, item) => acc + item.product, 0);
    const rawZ = sum + bias;
    const zFinal = hovered.layer === 1 ? nnForward.z1[hovered.index] : hovered.layer === 2 ? nnForward.z2[hovered.index] : nnForward.z3[hovered.index];
    const shift = zFinal - rawZ;
    const output = hovered.layer === 1 ? nnForward.a1[hovered.index] : hovered.layer === 2 ? nnForward.a2[hovered.index] : nnForward.probs[hovered.index];
    const rows = products
      .slice(0, 4)
      .map((item, idx) => `In${idx + 1}: ${item.value.toFixed(2)} × ${item.weight.toFixed(2)} = ${item.product.toFixed(2)}`);
    rows.push(`Bias: +${bias.toFixed(2)}`);
    if (Math.abs(shift) > 1e-3) rows.push(`Train shift: ${shift >= 0 ? '+' : ''}${shift.toFixed(2)}`);
    rows.push(`z = ${zFinal.toFixed(2)}`);
    return {
      title: `${layerLabel} - Neuron ${hovered.index + 1}`,
      description: 'Weighted sum plus bias, then activation.',
      rows,
      output: output.toFixed(3),
    };
  }, [hovered, nnForward]);

  return (
    <div className="studio-god-nnscene" ref={containerRef}>
      <div className="studio-god-nnscene-grid" />
      <canvas
        ref={canvasRef}
        className="studio-god-nnscene-canvas"
        onPointerDown={() => onAnyClick()}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          setPointer({ x, y });
          const panX = panRef.current.x;
          const panY = panRef.current.y;
          const nearest = layout.nodes.find((node) => Math.hypot(x - panX - node.x, y - panY - node.y) <= 22) ?? null;
          setHovered(nearest);
          setTooltipPos({ x: x + 18, y });
        }}
        onMouseLeave={() => {
          setHovered(null);
          setPointer({ x: size.width * 0.5, y: size.height * 0.5 });
        }}
      />

      {showCheat ? (
        <aside className="studio-god-nnscene-cheat">
          <h4>How it works</h4>
          <p>Neurons multiply incoming data by weights, add bias, and pass through activation.</p>
          <ul>
            <li><strong>W</strong> Weight: connection strength</li>
            <li><strong>b</strong> Bias: neuron threshold</li>
            <li><strong>z</strong> Sum: (In x W) + b</li>
            <li><strong>σ</strong> Activation: maps value to output</li>
            <li><strong>δ</strong> Error signal for weight updates</li>
          </ul>
        </aside>
      ) : null}

      <div
        className={`studio-god-nnscene-float studio-god-nnscene-float-top ${phaseMeta.forward ? 'is-forward' : 'is-backprop'}`}
        style={{ left: `${clamp(packet.x, topMinX, topMaxX)}px` }}
      >
        <p className="studio-god-nnscene-phase">{phaseMeta.label}</p>
        <p className="studio-god-nnscene-english">{phaseMeta.english}</p>
        <p className="studio-god-nnscene-math">{phaseMeta.math}</p>
      </div>

      <div
        className={`studio-god-nnscene-float studio-god-nnscene-float-bottom ${phaseMeta.forward ? 'is-forward' : 'is-backprop'}`}
        style={{ left: `${clamp(packet.x, bottomMinX, bottomMaxX)}px` }}
      >
        <p className="studio-god-nnscene-example-head">Concrete Example (Top Neuron)</p>
        <pre className="studio-god-nnscene-example">{phaseMeta.example}</pre>
      </div>

      <div
        className={`studio-god-nnscene-tooltip ${hovered ? 'is-visible' : ''}`}
        style={{ left: `${Math.min(size.width - 286, Math.max(10, tooltipPos.x))}px`, top: `${Math.min(size.height - 16, Math.max(16, tooltipPos.y))}px` }}
      >
        {tooltipData ? (
          <>
            <h5>{tooltipData.title}</h5>
            <p>{tooltipData.description}</p>
            <div className="studio-god-nnscene-tooltip-calc">
              {tooltipData.rows.map((row, index) => (
                <span key={`tt-row-${index}`}>{row}</span>
              ))}
            </div>
            <div className="studio-god-nnscene-tooltip-out">Output: {tooltipData.output}</div>
          </>
        ) : null}
      </div>
    </div>
  );
}

export function TextbookLearnMode() {
  const [state, setState] = useState<LearnState>(initialState());
  const stateRef = useRef(state);

  const shellRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const tooltipTitleRef = useRef<HTMLHeadingElement | null>(null);
  const tooltipBodyRef = useRef<HTMLDivElement | null>(null);

  const cnnRootRef = useRef<HTMLDivElement | null>(null);
  const inputGridRef = useRef<HTMLDivElement | null>(null);
  const convGridRef = useRef<HTMLDivElement | null>(null);
  const actGridRef = useRef<HTMLDivElement | null>(null);
  const poolGridRef = useRef<HTMLDivElement | null>(null);
  const flattenTargetsRef = useRef<HTMLDivElement | null>(null);
  const scanWindowRef = useRef<HTMLDivElement | null>(null);
  const frustumSvgRef = useRef<SVGSVGElement | null>(null);
  const frustumGroupRef = useRef<SVGGElement | null>(null);
  const frustumFacesRef = useRef<SVGPolygonElement[]>([]);
  const unzipOverlayRef = useRef<HTMLDivElement | null>(null);
  const poolSourceGeomRef = useRef<CellBox[] | null>(null);

  const stageRefs = useRef<Record<CnnStage, HTMLElement | null>>({
    input: null,
    conv1: null,
    act1: null,
    pool1: null,
    flatten: null,
    dense: null,
    output: null,
  });

  const nnForward = useMemo(
    () => computeNnForward(state.nn.trainShift, state.nn.logitScale),
    [state.nn.logitScale, state.nn.trainShift]
  );

  const dims = useMemo(() => cnnDims(state.kernelDim), [state.kernelDim]);
  const kernel1 = useMemo(() => buildVerticalKernel(state.kernelDim), [state.kernelDim]);

  const conv1Raw = useMemo(() => convValid(INPUT_GRID, kernel1), [kernel1]);
  const conv1Act = useMemo(() => conv1Raw.map((row) => row.map(relu)), [conv1Raw]);
  const pool1 = useMemo(() => maxPool(conv1Act, 2), [conv1Act]);
  const flat = useMemo(() => pool1.flat(), [pool1]);
  const denseOut = useMemo(() => {
    const norm = flat.map((value) => value / 255);
    const stride = Math.max(1, Math.floor(norm.length / 3));
    const denseIn3 = [
      norm.slice(0, stride).reduce((acc, value) => acc + value, 0) / Math.max(stride, 1),
      norm.slice(stride, stride * 2).reduce((acc, value) => acc + value, 0) / Math.max(stride, 1),
      norm.slice(stride * 2).reduce((acc, value) => acc + value, 0) / Math.max(norm.length - stride * 2, 1),
    ];
    const hidden2 = [
      relu(0.72 * denseIn3[0] + 0.21 * denseIn3[1] - 0.18 * denseIn3[2] + 0.1),
      relu(-0.31 * denseIn3[0] + 0.64 * denseIn3[1] + 0.36 * denseIn3[2] - 0.04),
    ];
    const logit = 1.18 * hidden2[0] - 0.74 * hidden2[1] + 0.08;
    const probSmile = 1 / (1 + Math.exp(-logit));
    return {
      denseIn3,
      hidden2,
      logit,
      probSmile: clamp(probSmile, 0, 1),
      probFrown: clamp(1 - probSmile, 0, 1),
    };
  }, [flat]);

  useEffect(() => {
    stateRef.current = state;
    syncTooltip();
    syncScanWindowAndFrustum();
    syncUnzipOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  useEffect(() => {
    let rafId = 0;
    let last = performance.now();

    const frame = (now: number) => {
      const dt = now - last;
      last = now;
      setState((prev) => tickState(prev, dt));
      rafId = window.requestAnimationFrame(frame);
    };

    const handleResize = () => {
      syncTooltip();
      syncScanWindowAndFrustum();
      syncUnzipOverlay();
    };

    rafId = window.requestAnimationFrame(frame);
    window.addEventListener('resize', handleResize);

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener('resize', handleResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const dispatchReset = () => {
    poolSourceGeomRef.current = null;
    if (state.mode === 'nn') {
      setState((prev) => ({
        ...prev,
        playback: 'idle',
        waitMs: 0,
        nn: buildNnInit(prev.nnTargetClass),
      }));
      return;
    }

    setState((prev) => ({
      ...prev,
      playback: 'idle',
      waitMs: 0,
      cnn: {
        stageIndex: 0,
        conv1Scan: 0,
        stageProgress: 1,
        densePass: 0,
      },
    }));
  };

  const dispatchStep = () => {
    setState((prev) => {
      const source = prev.mode === 'nn' ? { ...prev, nn: hideNnCheat(prev.nn) } : prev;
      if (source.mode === 'nn') {
        const phase = NN_PHASES[source.nn.phaseIndex];
        if (phase === 'output' && source.nn.stopAtOutput && source.nn.phaseProgress >= 1) {
          return {
            ...source,
            playback: 'paused',
            waitMs: 0,
          };
        }
        if (phase === 'update') {
          if (source.nn.phaseProgress < 1) {
            return {
              ...source,
              waitMs: 0,
              nn: {
                ...source.nn,
                phaseProgress: 1,
              },
            };
          }
          const trained = applyBackpropStep(
            source.nn.trainShift,
            source.nn.logitScale,
            source.learningRate,
            source.nnTargetClass,
            source.nnWeightDecay
          );
          return {
            ...source,
            waitMs: 0,
            nn: {
              ...source.nn,
              trainShift: trained.nextShift,
              logitScale: trained.nextLogitScale,
              trainStep: source.nn.trainStep + 1,
              prevPhaseIndex: source.nn.phaseIndex,
              phaseIndex: NN_PHASES.indexOf('input'),
              phaseProgress: 0,
              stopAtOutput: true,
            },
          };
        }
      }

      const next = source.mode === 'nn' ? advanceNn(source) : advanceCnn(source);
      return {
        ...next,
        playback: source.playback,
        waitMs: 0,
      };
    });
  };

  const dispatchPlayPause = () => {
    setState((prev) => {
      const source = prev.mode === 'nn' ? { ...prev, nn: hideNnCheat(prev.nn) } : prev;
      if (
        source.mode === 'nn' &&
        source.playback !== 'playing' &&
        source.nn.stopAtOutput &&
        NN_PHASES[source.nn.phaseIndex] === 'output' &&
        source.nn.phaseProgress >= 1
      ) {
        const nextNn = buildNnInit(source.nnTargetClass);
        return {
          ...source,
          playback: 'playing',
          waitMs: 0,
          nn: {
            ...nextNn,
            phaseProgress: 0,
          },
        };
      }
      return {
        ...source,
        playback: source.playback === 'playing' ? 'paused' : 'playing',
        waitMs: 0,
      };
    });
  };

  const dispatchSkip = () => {
    setState((prev) => {
      const source = prev.mode === 'nn' ? { ...prev, nn: hideNnCheat(prev.nn) } : prev;
      if (source.mode === 'nn') {
        const phase = NN_PHASES[source.nn.phaseIndex];
        if (phase === 'output' && source.nn.stopAtOutput && source.nn.phaseProgress >= 1) {
          return {
            ...source,
            playback: 'paused',
            waitMs: 0,
          };
        }
        if (phase === 'update') {
          const trained = applyBackpropStep(
            source.nn.trainShift,
            source.nn.logitScale,
            source.learningRate,
            source.nnTargetClass,
            source.nnWeightDecay
          );
          return {
            ...source,
            playback: source.playback,
            waitMs: 0,
            nn: {
              ...source.nn,
              trainShift: trained.nextShift,
              logitScale: trained.nextLogitScale,
              trainStep: source.nn.trainStep + 1,
              prevPhaseIndex: source.nn.phaseIndex,
              phaseIndex: NN_PHASES.indexOf('input'),
              phaseProgress: 1,
              stopAtOutput: true,
            },
          };
        }
        return {
          ...source,
          playback: source.playback,
          waitMs: 0,
          nn: { ...advanceNn(source).nn, phaseProgress: 1 },
        };
      }

      const next = skipCnnLayer(source);
      return { ...next, playback: source.playback, waitMs: 0 };
    });
  };

  const setMode = (mode: LearnMode) => {
    poolSourceGeomRef.current = null;
    setState((prev) => ({
      ...prev,
      mode,
      playback: 'idle',
      waitMs: 0,
      nn: mode === 'nn' ? buildNnInit(prev.nnTargetClass) : prev.nn,
      cnn:
        mode === 'cnn'
          ? {
              stageIndex: 0,
              conv1Scan: 0,
              stageProgress: 1,
              densePass: 0,
            }
          : prev.cnn,
    }));
  };

  const syncTooltip = () => {
    const shell = shellRef.current;
    const tooltip = tooltipRef.current;
    const title = tooltipTitleRef.current;
    const body = tooltipBodyRef.current;
    if (!shell || !tooltip || !title || !body) return;

    const current = stateRef.current;
    if (current.mode === 'nn') {
      tooltip.style.opacity = '0';
      tooltip.classList.remove('is-cnn-docked');
      tooltip.classList.remove('is-pulse-green');
      return;
    }

    tooltip.style.opacity = '0';
    tooltip.classList.remove('is-cnn-docked');
    tooltip.classList.remove('is-pulse-green');
    return;
  };

  const syncScanWindowAndFrustum = () => {
    const current = stateRef.current;
    const root = cnnRootRef.current;
    const svg = frustumSvgRef.current;
    const group = frustumGroupRef.current;
    const windowEl = scanWindowRef.current;
    if (!root || !svg || !group || !windowEl) return;

    const rootRect = root.getBoundingClientRect();
    const stage = CNN_STAGES[current.cnn.stageIndex];

    svg.setAttribute('viewBox', `0 0 ${Math.max(1, rootRect.width)} ${Math.max(1, rootRect.height)}`);

    const hideFrustum = () => {
      windowEl.style.opacity = '0';
      frustumFacesRef.current.forEach((face) => face.setAttribute('points', ''));
    };

    const ensureFaces = () => {
      if (frustumFacesRef.current.length === 4) return frustumFacesRef.current;
      frustumFacesRef.current = [];
      group.replaceChildren();
      for (let i = 0; i < 4; i += 1) {
        const face = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        face.setAttribute('class', 'studio-god-frustum-face');
        group.appendChild(face);
        frustumFacesRef.current.push(face);
      }
      return frustumFacesRef.current;
    };

    const makeFrustum = (
      sourceBoxes: CellBox[],
      targetGrid: HTMLElement | null,
      scanIndex: number,
      sourceWidth: number,
      windowSize: number,
      targetWidth: number
    ) => {
      if (!targetGrid) {
        hideFrustum();
        return;
      }

      const scan = mapScan(scanIndex, targetWidth);
      const startIndex = scan.row * sourceWidth + scan.col;
      const endIndex = (scan.row + windowSize - 1) * sourceWidth + (scan.col + windowSize - 1);
      const startCell = sourceBoxes[startIndex];
      const endCell = sourceBoxes[endIndex];

      if (!startCell || !endCell) {
        hideFrustum();
        return;
      }

      const left = startCell.left;
      const top = startCell.top;
      const widthPx = endCell.left + endCell.width - startCell.left;
      const heightPx = endCell.top + endCell.height - startCell.top;

      windowEl.style.width = `${widthPx}px`;
      windowEl.style.height = `${heightPx}px`;
      windowEl.style.left = `${left}px`;
      windowEl.style.top = `${top}px`;
      windowEl.style.transform = 'translate3d(0, 0, 0)';
      windowEl.style.opacity = '1';
      windowEl.style.borderColor = stage === 'conv1' ? 'rgba(245, 158, 11, 0.95)' : 'rgba(16, 185, 129, 0.95)';
      windowEl.style.background = stage === 'conv1' ? 'rgba(245, 158, 11, 0.18)' : 'rgba(16, 185, 129, 0.18)';

      const targetCellIndex = scan.row * targetWidth + scan.col;
      const targetCell = targetGrid.children[targetCellIndex] as HTMLElement | undefined;
      if (!targetCell) {
        hideFrustum();
        return;
      }

      const targetRect = targetCell.getBoundingClientRect();
      const target = {
        x: targetRect.left - rootRect.left + targetRect.width * 0.5,
        y: targetRect.top - rootRect.top + targetRect.height * 0.5,
      };

      const corners = [
        { x: left, y: top },
        { x: left + widthPx, y: top },
        { x: left + widthPx, y: top + heightPx },
        { x: left, y: top + heightPx },
      ];

      const faces = ensureFaces();
      corners.forEach((corner, index) => {
        const next = corners[(index + 1) % corners.length];
        faces[index].setAttribute('points', `${corner.x},${corner.y} ${next.x},${next.y} ${target.x},${target.y}`);
      });
    };

    if (stage === 'pool1' && poolGridRef.current) {
      poolSourceGeomRef.current = cellBoxesFromGrid(poolGridRef.current, rootRect);
    }

    if (stage === 'conv1' && inputGridRef.current && convGridRef.current) {
      makeFrustum(cellBoxesFromGrid(inputGridRef.current, rootRect), convGridRef.current, current.cnn.conv1Scan, dims.input, dims.kernel1, dims.conv1);
      return;
    }

    hideFrustum();
  };

  const syncUnzipOverlay = () => {
    const root = cnnRootRef.current;
    const overlay = unzipOverlayRef.current;
    const targets = flattenTargetsRef.current;
    if (!root || !overlay || !targets) return;

    const current = stateRef.current;
    const rootRect = root.getBoundingClientRect();
    const sourceBoxes = poolGridRef.current ? cellBoxesFromGrid(poolGridRef.current, rootRect) : poolSourceGeomRef.current;
    if (!sourceBoxes || sourceBoxes.length === 0) return;
    const targetCells = Array.from(targets.children) as HTMLElement[];
    const overlayCells = Array.from(overlay.children) as HTMLElement[];

    overlayCells.forEach((cell, index) => {
      const source = sourceBoxes[index];
      const target = targetCells[index];
      if (!source || !target) return;

      const targetRect = target.getBoundingClientRect();
      const targetLeft = targetRect.left - rootRect.left;
      const targetTop = targetRect.top - rootRect.top;
      const left = source.left;
      const top = source.top;
      const dx = targetLeft - left;
      const dy = targetTop - top;

      cell.style.left = `${left}px`;
      cell.style.top = `${top}px`;
      cell.style.width = `${source.width}px`;
      cell.style.height = `${source.height}px`;
      cell.style.setProperty('--ux', `${dx}px`);
      cell.style.setProperty('--uy', `${dy}px`);

      const nowStage = CNN_STAGES[current.cnn.stageIndex];
      const show = nowStage === 'flatten';
      const progress = nowStage === 'flatten' ? current.cnn.stageProgress : 0;
      cell.style.transform = `translate(${dx * progress}px, ${dy * progress}px)`;

      cell.classList.toggle('is-show', show);
      cell.classList.remove('is-column');
    });
  };

  const nnCurrentPhase = NN_PHASES[state.nn.phaseIndex];

  const showNn = state.mode === 'nn';
  const showNnCheat =
    showNn &&
    state.nn.cheatHiddenUntilStep === null &&
    state.playback !== 'playing' &&
    state.nn.phaseProgress >= 1;
  const hideNnCheatOnAnyClick = () => {
    setState((prev) => {
      if (prev.mode !== 'nn') return prev;
      return {
        ...prev,
        nn: hideNnCheat(prev.nn),
      };
    });
  };
  const cnnStage = CNN_STAGES[state.cnn.stageIndex];
  const cnnStageOrder: CnnStage[] = ['input', 'conv1', 'act1', 'pool1', 'flatten', 'dense', 'output'];
  const currentStageOrder = cnnStageOrder.indexOf(cnnStage);
  const isStageReached = (stage: CnnStage): boolean => cnnStageOrder.indexOf(stage) <= currentStageOrder;
  const cnnFlowItems: Array<{ label: string; stage: CnnStage }> = [
    { label: 'Input', stage: 'input' },
    { label: 'Conv1', stage: 'conv1' },
    { label: 'Activation', stage: 'act1' },
    { label: 'MaxPool', stage: 'pool1' },
    { label: 'Flatten', stage: 'flatten' },
    { label: 'Dense 3->2->1', stage: 'dense' },
    { label: 'Output', stage: 'output' },
  ];

  const convDisplay = conv1Raw;
  const actDisplay = conv1Act;
  const poolDisplay = pool1;
  const flatDisplay = flat.slice(0, Math.min(8, flat.length));
  const convScan = mapScan(state.cnn.conv1Scan, dims.conv1);
  const convScore = dotPatch(INPUT_GRID, kernel1, convScan.row, convScan.col);

  const convVisibleIndex = isStageReached('conv1') ? (cnnStage === 'conv1' ? state.cnn.conv1Scan : 1000) : -1;

  const drawMatrixCells = (matrix: number[][], keyPrefix: string, visibleIndex = 1000, showNumbers = false, stageReady = true) => {
    const { min, max } = matrixRange(matrix);
    const cols = matrix[0]?.length ?? 1;
    const fullVisible = visibleIndex >= cols * Math.max(matrix.length, 1);
    return matrix.flatMap((row, rowIndex) =>
      row.map((value, colIndex) => {
        const index = rowIndex * cols + colIndex;
        const order = serpentineOrderIndex(rowIndex, colIndex, cols);
        const isVisible = stageReady && (fullVisible || order <= visibleIndex);
        const bg = heatColor(value, min, max);
        const fg = heatTextColor(value, min, max);
        return (
          <span
            key={`${keyPrefix}-${index}`}
            className={isVisible ? '' : 'is-dim'}
            style={{ background: bg, color: fg }}
            title={`(${rowIndex}, ${colIndex}) = ${formatCellValue(value)}`}
          >
            {showNumbers ? formatCellValue(value) : ''}
          </span>
        );
      })
    );
  };

  const denseEdge1Progress =
    cnnStage === 'dense' ? (state.cnn.densePass === 0 ? state.cnn.stageProgress : 1) : cnnStage === 'output' ? 1 : 0;
  const denseEdge2Progress =
    cnnStage === 'dense' ? (state.cnn.densePass === 0 ? 0 : state.cnn.stageProgress) : cnnStage === 'output' ? 1 : 0;
  const playbackSpeed = clamp(Number((1200 / state.tempoMs).toFixed(2)), 1, 24);
  const flattenRevealCount =
    cnnStage === 'flatten'
      ? Math.floor(state.cnn.stageProgress * flatDisplay.length)
      : isStageReached('dense') || isStageReached('output')
        ? flatDisplay.length
        : 0;

  return (
    <section className="studio-god-shell" ref={shellRef}>
      <header className="studio-god-toolbar">
        <div className="studio-god-toggle" role="tablist" aria-label="Learn mode tabs">
          <button type="button" className={`studio-god-tab ${showNn ? 'is-active' : ''}`} onClick={() => setMode('nn')}>
            Neural Network
          </button>
          <button type="button" className={`studio-god-tab ${!showNn ? 'is-active' : ''}`} onClick={() => setMode('cnn')}>
            Convolutional Pipeline
          </button>
        </div>

        <div className="studio-god-controls" role="group" aria-label="Playback controls">
          <button type="button" className="studio-god-btn" onClick={dispatchReset}>
            |&lt; Reset
          </button>
          <button type="button" className="studio-god-btn" onClick={dispatchStep}>
            Step
          </button>
          <button type="button" className="studio-god-btn" onClick={dispatchSkip}>
            Skip
          </button>
          <button type="button" className="studio-god-btn" onClick={dispatchPlayPause}>
            {state.playback === 'playing' ? 'Pause' : 'Play'}
          </button>
        </div>

        <div className="studio-god-hyperparams">
          {showNn ? (
            <>
              <label className="studio-god-param">
                <span>LR {state.learningRate.toFixed(2)}</span>
                <input
                  type="range"
                  min={0.01}
                  max={0.6}
                  step={0.01}
                  value={state.learningRate}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setState((prev) => ({ ...prev, learningRate: clamp(value, 0.01, 0.6) }));
                  }}
                />
              </label>
              <label className="studio-god-param">
                <span>Decay λ {state.nnWeightDecay.toFixed(2)}</span>
                <input
                  type="range"
                  min={0}
                  max={0.2}
                  step={0.01}
                  value={state.nnWeightDecay}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setState((prev) => ({ ...prev, nnWeightDecay: clamp(value, 0, 0.2) }));
                  }}
                />
              </label>
              <div className="studio-god-param studio-god-param-target">
                <span>Target</span>
                <div className="studio-god-mini-toggle" role="group" aria-label="NN target class">
                  <button
                    type="button"
                    className={`studio-god-mini-btn ${state.nnTargetClass === 0 ? 'is-active' : ''}`}
                    onClick={() =>
                      setState((prev) => ({
                        ...prev,
                        nnTargetClass: 0,
                        playback: prev.mode === 'nn' ? 'idle' : prev.playback,
                        waitMs: 0,
                        nn: prev.mode === 'nn' ? buildNnInit(0) : prev.nn,
                      }))
                    }
                  >
                    Class 0
                  </button>
                  <button
                    type="button"
                    className={`studio-god-mini-btn ${state.nnTargetClass === 1 ? 'is-active' : ''}`}
                    onClick={() =>
                      setState((prev) => ({
                        ...prev,
                        nnTargetClass: 1,
                        playback: prev.mode === 'nn' ? 'idle' : prev.playback,
                        waitMs: 0,
                        nn: prev.mode === 'nn' ? buildNnInit(1) : prev.nn,
                      }))
                    }
                  >
                    Class 1
                  </button>
                </div>
              </div>
            </>
          ) : null}
          <label className="studio-god-param">
            <span>{showNn ? `Speed ${playbackSpeed.toFixed(2)}x` : `Conv Speed ${playbackSpeed.toFixed(2)}x`}</span>
            <input
              type="range"
              min={1}
              max={24}
              step={0.5}
              value={playbackSpeed}
              onChange={(event) => {
                const speed = Number(event.target.value);
                const nextTempo = clamp(1200 / Math.max(speed, 0.1), 40, 1400);
                setState((prev) => ({ ...prev, tempoMs: nextTempo }));
              }}
            />
          </label>
          {!showNn ? (
            <label className="studio-god-param">
              <span>Kernel {state.kernelDim}x{state.kernelDim}</span>
              <input
                type="range"
                min={3}
                max={5}
                step={2}
                value={state.kernelDim}
                onChange={(event) => {
                  const value: 3 | 5 = Number(event.target.value) === 5 ? 5 : 3;
                  poolSourceGeomRef.current = null;
                  setState((prev) => ({
                    ...prev,
                    kernelDim: value,
                    playback: prev.mode === 'cnn' ? 'idle' : prev.playback,
                    waitMs: 0,
                    cnn:
                      prev.mode === 'cnn'
                        ? {
                            stageIndex: 0,
                            conv1Scan: 0,
                            stageProgress: 1,
                            densePass: 0,
                          }
                        : prev.cnn,
                  }));
                }}
              />
            </label>
          ) : null}
        </div>
      </header>

      <div className="studio-god-canvas">
        <section className={`studio-god-module ${showNn ? '' : 'is-hidden'}`}>
          {showNn ? (
            <GodNnCanvasStage
              phase={nnCurrentPhase}
              phaseProgress={state.nn.phaseProgress}
              nnForward={nnForward}
              trainShift={state.nn.trainShift}
              logitScale={state.nn.logitScale}
              trainStep={state.nn.trainStep}
              learningRate={state.learningRate}
              targetClass={state.nnTargetClass}
              weightDecay={state.nnWeightDecay}
              showCheat={showNnCheat}
              onAnyClick={hideNnCheatOnAnyClick}
            />
          ) : null}
        </section>

        <section className={`studio-god-module ${showNn ? 'is-hidden' : ''}`}>
          <h3 className="studio-god-module-title studio-god-module-flow" aria-label="CNN flow">
            {cnnFlowItems.map((item, index) => (
              <span key={`cnn-flow-${item.stage}`} className="studio-god-flow-step">
                <span className={`studio-god-flow-pill ${cnnStage === item.stage ? 'is-active' : isStageReached(item.stage) ? 'is-reached' : ''}`}>
                  {item.label}
                </span>
                {index < cnnFlowItems.length - 1 ? <span className="studio-god-flow-arrow">→</span> : null}
              </span>
            ))}
          </h3>

          <div className="studio-god-cnn-root" ref={cnnRootRef}>
            <svg className="studio-god-frustum-svg" ref={frustumSvgRef} preserveAspectRatio="none">
              <g ref={frustumGroupRef} />
            </svg>
            <div className="studio-god-scan-window" ref={scanWindowRef} />

            <div className="studio-god-cnn-perspective">
              <article
                className={`studio-god-layer-card studio-god-card-input ${cnnStage === 'input' ? 'is-active' : ''}`}
                ref={(node) => {
                  stageRefs.current.input = node;
                }}
              >
                <p>Input</p>
                <div className="studio-god-layer-grid" style={gridStyle(INPUT_GRID[0].length)} ref={inputGridRef}>
                  {drawMatrixCells(INPUT_GRID, 'in')}
                </div>
                <div className="studio-god-stage-tile">
                  <strong>Input tensor:</strong> 28x28 grayscale smiley.
                  <br />
                  Values are intensity pixels in [0, 255] used as raw features.
                  <br />
                  Brighter values indicate stronger local signal energy.
                </div>
              </article>

              {isStageReached('conv1') ? (
                <article
                  className={`studio-god-layer-card studio-god-card-conv ${cnnStage === 'conv1' ? 'is-active' : 'is-ready'}`}
                  ref={(node) => {
                    stageRefs.current.conv1 = node;
                  }}
                >
                  <p>Conv Layer 1</p>
                  <div className="studio-god-layer-grid" style={gridStyle(convDisplay[0].length)} ref={convGridRef}>
                    {drawMatrixCells(convDisplay, 'conv1', convVisibleIndex, false, true)}
                  </div>
                  <div className="studio-god-stage-tile">
                    <strong>{`Patch (${convScan.row}, ${convScan.col})`}</strong>
                    <br />
                    Kernel is vertical-edge detector: [-1, 0, 1] rows.
                    <br />
                    {`sum(window * kernel) = ${convScore.toFixed(2)} (high magnitude => strong edge)`}
                  </div>
                </article>
              ) : null}

              {isStageReached('act1') ? (
                <article
                  className={`studio-god-layer-card studio-god-card-act ${cnnStage === 'act1' ? 'is-active' : 'is-ready'}`}
                  ref={(node) => {
                    stageRefs.current.act1 = node;
                  }}
                >
                  <p>Activation</p>
                  <div className="studio-god-layer-grid" style={gridStyle(actDisplay[0].length)} ref={actGridRef}>
                    {drawMatrixCells(actDisplay, 'act1', 1000, false, true)}
                  </div>
                  <div className="studio-god-stage-tile">
                    <strong>Activation:</strong> a1 = ReLU(conv1)
                    <br />
                    ReLU(x) = max(0, x), so negative responses become 0.
                    <br />
                    This keeps only positive vertical-edge evidence for later layers.
                  </div>
                </article>
              ) : null}

              {isStageReached('pool1') ? (
                <article
                  className={`studio-god-layer-card studio-god-card-pool ${cnnStage === 'pool1' ? 'is-active' : 'is-ready'}`}
                  ref={(node) => {
                    stageRefs.current.pool1 = node;
                  }}
                >
                  <p>Max Pooling</p>
                  <div className="studio-god-layer-grid" style={gridStyle(poolDisplay[0].length)} ref={poolGridRef}>
                    {drawMatrixCells(poolDisplay, 'pool1', 1000, false, true)}
                  </div>
                  <div className="studio-god-stage-tile">
                    <strong>MaxPool 2x2:</strong> keeps local maxima.
                    <br />
                    Each pooled cell = max of a 2x2 neighborhood.
                    <br />
                    Result: fewer cells, better translation tolerance, less noise.
                  </div>
                </article>
              ) : null}

              {isStageReached('flatten') ? (
                <article
                  className={`studio-god-layer-card studio-god-card-flatten ${cnnStage === 'flatten' ? 'is-active' : 'is-ready'}`}
                  ref={(node) => {
                    stageRefs.current.flatten = node;
                  }}
                >
                  <p>Flatten</p>
                  <div className="studio-god-flat-targets" ref={flattenTargetsRef}>
                    {flatDisplay.map((value, index) => (
                      <span key={`flat-${index}`} className={index < flattenRevealCount ? 'is-show' : ''}>
                        {index < flattenRevealCount ? value.toFixed(1) : ''}
                      </span>
                    ))}
                  </div>
                  <div className="studio-god-stage-tile">
                    <strong>Flatten:</strong> 2D map -&gt; 1D vector.
                    <br />
                    Spatial structure is serialized into feature order.
                    <br />
                    This vector feeds the classifier as numeric evidence.
                  </div>
                </article>
              ) : null}

              {isStageReached('dense') ? (
                <article
                  className={`studio-god-layer-card studio-god-card-dense ${cnnStage === 'dense' ? 'is-active' : 'is-ready'}`}
                  ref={(node) => {
                    stageRefs.current.dense = node;
                  }}
                >
                  <p>Dense Layer (3-&gt;2-&gt;1)</p>
                  <svg viewBox="0 0 180 150" className="studio-god-dense-svg">
                    {[30, 75, 120].map((y, index) => (
                      <circle key={`d-in-${index}`} cx="26" cy={y} r="7" className="studio-god-dense-node is-in" />
                    ))}
                    {[46, 104].map((y, index) => (
                      <circle key={`d-h-${index}`} cx="92" cy={y} r="7" className="studio-god-dense-node is-hidden" />
                    ))}
                    <circle cx="158" cy="75" r="8.5" className="studio-god-dense-node is-out" />

                    {[30, 75, 120].map((sy, i) =>
                      [46, 104].map((ey, j) => {
                        const x2 = 26 + (92 - 26) * denseEdge1Progress;
                        const y2 = sy + (ey - sy) * denseEdge1Progress;
                        return <line key={`d-l1-${i}-${j}`} x1="26" y1={sy} x2={x2} y2={y2} className="studio-god-dense-edge" />;
                      })
                    )}

                    {[46, 104].map((sy, i) => {
                      const ey = 75;
                      const x2 = 92 + (158 - 92) * denseEdge2Progress;
                      const y2 = sy + (ey - sy) * denseEdge2Progress;
                      return <line key={`d-l2-${i}`} x1="92" y1={sy} x2={x2} y2={y2} className="studio-god-dense-edge" />;
                    })}
                  </svg>
                  <div className="studio-god-stage-tile">
                    <strong>Dense projection:</strong> 3 -&gt; 2 -&gt; 1
                    <br />
                    {`Input3 [${denseOut.denseIn3.map((value) => value.toFixed(2)).join(', ')}]`}
                    <br />
                    {`Hidden2 [${denseOut.hidden2.map((value) => value.toFixed(2)).join(', ')}], sigmoid=${denseOut.probSmile.toFixed(3)}`}
                    <br />
                    Weighted sums learn non-linear combinations of pooled features.
                  </div>
                </article>
              ) : null}

              {isStageReached('output') ? (
                <article
                  className={`studio-god-layer-card studio-god-card-output ${cnnStage === 'output' ? 'is-active' : 'is-ready'}`}
                  ref={(node) => {
                    stageRefs.current.output = node;
                  }}
                >
                  <p>Output (1 Neuron)</p>
                  <div className="studio-god-output-emoji">{denseOut.probSmile >= 0.5 ? '🙂' : '🙁'}</div>
                  <div className="studio-god-dense-readout">
                    <span className={denseOut.probSmile >= 0.5 ? 'is-win is-smile' : 'is-smile'}>
                      {`🙂 Smile ${(denseOut.probSmile * 100).toFixed(1)}%`}
                    </span>
                    <span className={denseOut.probSmile < 0.5 ? 'is-win is-frown' : 'is-frown'}>
                      {`🙁 Frown ${(denseOut.probFrown * 100).toFixed(1)}%`}
                    </span>
                    <span>{`Final z = ${denseOut.logit.toFixed(3)} | y = σ(z)`}</span>
                  </div>
                  <div className="studio-god-stage-tile">
                    <strong>Decision rule:</strong> if y &gt;= 0.5, classify Smile; else Frown.
                    <br />
                    Confidence reflects how strongly features support each class.
                  </div>
                </article>
              ) : null}
            </div>

            <div className="studio-god-unzip-overlay" ref={unzipOverlayRef}>
              {flatDisplay.map((value, index) => (
                <span key={`zip-${index}`} aria-label={`flatten-${index}-${value.toFixed(1)}`} />
              ))}
            </div>
          </div>
        </section>

        <aside className="studio-god-tooltip" ref={tooltipRef}>
          <h4 ref={tooltipTitleRef}>Math HUD</h4>
          <div className="studio-god-tooltip-body" ref={tooltipBodyRef}>
            State-aware formula
          </div>
        </aside>
      </div>
    </section>
  );
}

