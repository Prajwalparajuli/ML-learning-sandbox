import type { DataPoint, DatasetType, EvaluationMode, ModelParams, ModelType, RandomDataRecipe } from '../store/modelStore';

interface LinearSolution {
  intercept: number;
  coefficients: number[];
}

export interface RegressionFit {
  modelType: ModelType;
  intercept: number;
  coefficients: number[];
  featurePowers: number[];
  predict: (input: number | number[]) => number;
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(items: T[], seed: number): T[] {
  const random = createSeededRandom(seed);
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

function softThreshold(value: number, lambda: number): number {
  if (value > lambda) return value - lambda;
  if (value < -lambda) return value + lambda;
  return 0;
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function rbfKernel(a: number[], b: number[], gamma: number): number {
  let distanceSq = 0;
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; i++) {
    const delta = (a[i] ?? 0) - (b[i] ?? 0);
    distanceSq += delta * delta;
  }
  return Math.exp(-gamma * distanceSq);
}

function isClassificationModel(modelType: ModelType): boolean {
  return modelType === 'logistic_classifier'
    || modelType === 'knn_classifier'
    || modelType === 'svm_classifier'
    || modelType === 'decision_tree_classifier'
    || modelType === 'random_forest_classifier'
    || modelType === 'adaboost_classifier'
    || modelType === 'gradient_boosting_classifier';
}

function isClassificationDataset(dataset: DatasetType): boolean {
  return dataset === 'class_linear' || dataset === 'class_overlap' || dataset === 'class_moons' || dataset === 'class_imbalanced';
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] {
  const n = vector.length;
  const a = matrix.map((row) => [...row]);
  const b = [...vector];

  for (let pivot = 0; pivot < n; pivot++) {
    let maxRow = pivot;
    for (let row = pivot + 1; row < n; row++) {
      if (Math.abs(a[row][pivot]) > Math.abs(a[maxRow][pivot])) {
        maxRow = row;
      }
    }

    if (Math.abs(a[maxRow][pivot]) < 1e-12) {
      continue;
    }

    if (maxRow !== pivot) {
      [a[pivot], a[maxRow]] = [a[maxRow], a[pivot]];
      [b[pivot], b[maxRow]] = [b[maxRow], b[pivot]];
    }

    const pivotValue = a[pivot][pivot];
    for (let col = pivot; col < n; col++) {
      a[pivot][col] /= pivotValue;
    }
    b[pivot] /= pivotValue;

    for (let row = 0; row < n; row++) {
      if (row === pivot) continue;
      const factor = a[row][pivot];
      if (Math.abs(factor) < 1e-12) continue;
      for (let col = pivot; col < n; col++) {
        a[row][col] -= factor * a[pivot][col];
      }
      b[row] -= factor * b[pivot];
    }
  }

  return b;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function getPointFeatures(point: DataPoint): number[] {
  if (Array.isArray(point.features) && point.features.length > 0) return point.features;
  if (typeof point.x2 === 'number') return [point.x, point.x2];
  return [point.x];
}

function toFeatureArray(input: number | number[]): number[] {
  return Array.isArray(input) ? input : [input];
}

function getFeaturePowers(x: number, powers: number[]): number[] {
  return powers.map((power) => Math.pow(x, power));
}

function fitLeastSquaresWithPowers(data: DataPoint[], powers: number[], ridgeAlpha = 0): LinearSolution {
  if (data.length === 0 || powers.length === 0) {
    return { intercept: 0, coefficients: [] };
  }

  const yValues = data.map((point) => point.y);
  const yMean = mean(yValues);
  const xMatrix = data.map((point) => getFeaturePowers(point.x, powers));
  const xMeans = powers.map((_, col) => mean(xMatrix.map((row) => row[col])));
  const centeredX = xMatrix.map((row) => row.map((value, col) => value - xMeans[col]));
  const centeredY = yValues.map((value) => value - yMean);

  const p = powers.length;
  const xtx = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  const xty = Array.from({ length: p }, () => 0);

  for (let row = 0; row < centeredX.length; row++) {
    for (let colA = 0; colA < p; colA++) {
      xty[colA] += centeredX[row][colA] * centeredY[row];
      for (let colB = 0; colB < p; colB++) {
        xtx[colA][colB] += centeredX[row][colA] * centeredX[row][colB];
      }
    }
  }

  for (let i = 0; i < p; i++) {
    xtx[i][i] += ridgeAlpha * data.length;
  }

  const coefficients = solveLinearSystem(xtx, xty);
  const intercept = yMean - coefficients.reduce((sum, coefficient, index) => sum + coefficient * xMeans[index], 0);
  return { intercept, coefficients };
}

function fitCoordinateDescent(
  data: DataPoint[],
  powers: number[],
  alpha: number,
  l1Ratio: number
): LinearSolution {
  if (data.length === 0 || powers.length === 0) {
    return { intercept: 0, coefficients: [] };
  }

  const yValues = data.map((point) => point.y);
  const yMean = mean(yValues);
  const xMatrix = data.map((point) => getFeaturePowers(point.x, powers));
  const xMeans = powers.map((_, col) => mean(xMatrix.map((row) => row[col])));
  const centeredX = xMatrix.map((row) => row.map((value, col) => value - xMeans[col]));
  const centeredY = yValues.map((value) => value - yMean);
  const n = data.length;
  const p = powers.length;
  const beta = Array.from({ length: p }, () => 0);

  const maxIterations = 400;
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    let maxDelta = 0;
    for (let feature = 0; feature < p; feature++) {
      let rho = 0;
      let denominator = 0;

      for (let row = 0; row < n; row++) {
        let predictionWithoutFeature = 0;
        for (let col = 0; col < p; col++) {
          if (col !== feature) {
            predictionWithoutFeature += centeredX[row][col] * beta[col];
          }
        }
        const residual = centeredY[row] - predictionWithoutFeature;
        rho += centeredX[row][feature] * residual;
        denominator += centeredX[row][feature] * centeredX[row][feature];
      }

      const l1Penalty = alpha * l1Ratio * n;
      const l2Penalty = alpha * (1 - l1Ratio) * n;
      const newBeta = softThreshold(rho, l1Penalty) / (denominator + l2Penalty + 1e-12);
      maxDelta = Math.max(maxDelta, Math.abs(newBeta - beta[feature]));
      beta[feature] = newBeta;
    }

    if (maxDelta < 1e-6) {
      break;
    }
  }

  const intercept = yMean - beta.reduce((sum, coefficient, index) => sum + coefficient * xMeans[index], 0);
  return { intercept, coefficients: beta };
}

function fitLeastSquaresMulti(data: DataPoint[], ridgeAlpha = 0): LinearSolution {
  if (data.length === 0) return { intercept: 0, coefficients: [] };
  const xMatrix = data.map((point) => getPointFeatures(point));
  const p = xMatrix[0].length;
  const yValues = data.map((point) => point.y);
  const yMean = mean(yValues);
  const xMeans = Array.from({ length: p }, (_, col) => mean(xMatrix.map((row) => row[col])));
  const centeredX = xMatrix.map((row) => row.map((value, col) => value - xMeans[col]));
  const centeredY = yValues.map((value) => value - yMean);

  const xtx = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  const xty = Array.from({ length: p }, () => 0);

  for (let row = 0; row < centeredX.length; row++) {
    for (let colA = 0; colA < p; colA++) {
      xty[colA] += centeredX[row][colA] * centeredY[row];
      for (let colB = 0; colB < p; colB++) {
        xtx[colA][colB] += centeredX[row][colA] * centeredX[row][colB];
      }
    }
  }

  for (let feature = 0; feature < p; feature++) {
    xtx[feature][feature] += ridgeAlpha * data.length;
  }

  const coefficients = solveLinearSystem(xtx, xty);
  const intercept = yMean - coefficients.reduce((sum, coefficient, index) => sum + coefficient * xMeans[index], 0);
  return { intercept, coefficients };
}

function fitCoordinateDescentMulti(
  data: DataPoint[],
  alpha: number,
  l1Ratio: number
): LinearSolution {
  if (data.length === 0) return { intercept: 0, coefficients: [] };
  const xMatrix = data.map((point) => getPointFeatures(point));
  const p = xMatrix[0].length;
  const n = data.length;
  const yValues = data.map((point) => point.y);
  const yMean = mean(yValues);
  const xMeans = Array.from({ length: p }, (_, col) => mean(xMatrix.map((row) => row[col])));
  const centeredX = xMatrix.map((row) => row.map((value, col) => value - xMeans[col]));
  const centeredY = yValues.map((value) => value - yMean);
  const beta = Array.from({ length: p }, () => 0);
  const maxIterations = 500;

  for (let iteration = 0; iteration < maxIterations; iteration++) {
    let maxDelta = 0;
    for (let feature = 0; feature < p; feature++) {
      let rho = 0;
      let denominator = 0;

      for (let row = 0; row < n; row++) {
        let predictionWithoutFeature = 0;
        for (let col = 0; col < p; col++) {
          if (col !== feature) predictionWithoutFeature += centeredX[row][col] * beta[col];
        }
        const residual = centeredY[row] - predictionWithoutFeature;
        rho += centeredX[row][feature] * residual;
        denominator += centeredX[row][feature] * centeredX[row][feature];
      }

      const l1Penalty = alpha * l1Ratio * n;
      const l2Penalty = alpha * (1 - l1Ratio) * n;
      const newBeta = softThreshold(rho, l1Penalty) / (denominator + l2Penalty + 1e-12);
      maxDelta = Math.max(maxDelta, Math.abs(newBeta - beta[feature]));
      beta[feature] = newBeta;
    }

    if (maxDelta < 1e-6) break;
  }

  const intercept = yMean - beta.reduce((sum, coefficient, index) => sum + coefficient * xMeans[index], 0);
  return { intercept, coefficients: beta };
}

function predictFromMultiSolution(input: number | number[], solution: LinearSolution): number {
  const features = toFeatureArray(input);
  let prediction = solution.intercept;
  for (let index = 0; index < solution.coefficients.length; index++) {
    prediction += solution.coefficients[index] * (features[index] ?? 0);
  }
  return prediction;
}

function predictFromLinearSolution(x: number, powers: number[], solution: LinearSolution): number {
  let prediction = solution.intercept;
  for (let i = 0; i < powers.length; i++) {
    prediction += solution.coefficients[i] * Math.pow(x, powers[i]);
  }
  return prediction;
}

function expandPolynomialFeatures(features: number[], degree: number): number[] {
  const dim = features.length;
  const terms: number[] = [];
  const clampedDegree = Math.max(1, Math.min(degree, 6));

  const build = (start: number, depth: number, product: number) => {
    if (depth > 0) terms.push(product);
    if (depth === clampedDegree) return;
    for (let i = start; i < dim; i++) {
      build(i, depth + 1, product * (features[i] ?? 0));
    }
  };

  build(0, 0, 1);
  return terms;
}

function covarianceMatrix(xMatrix: number[][]): number[][] {
  const n = Math.max(1, xMatrix.length);
  const p = xMatrix[0]?.length ?? 0;
  const means = Array.from({ length: p }, (_, j) => mean(xMatrix.map((row) => row[j])));
  const centered = xMatrix.map((row) => row.map((value, j) => value - means[j]));
  const cov = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  for (let i = 0; i < centered.length; i++) {
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        cov[a][b] += centered[i][a] * centered[i][b];
      }
    }
  }
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) cov[a][b] /= Math.max(n - 1, 1);
  }
  return cov;
}

function normalizeVector(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, value) => sum + value * value, 0));
  if (norm < 1e-10) return v.map(() => 0);
  return v.map((value) => value / norm);
}

function matVec(mat: number[][], vec: number[]): number[] {
  return mat.map((row) => row.reduce((sum, value, j) => sum + value * (vec[j] ?? 0), 0));
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, i) => sum + value * (b[i] ?? 0), 0);
}

function computePcaComponents(xMatrix: number[][], k: number): { components: number[][]; means: number[] } {
  const p = xMatrix[0]?.length ?? 0;
  const means = Array.from({ length: p }, (_, j) => mean(xMatrix.map((row) => row[j])));
  if (p === 0) return { components: [], means };
  let cov = covarianceMatrix(xMatrix);
  const components: number[][] = [];
  const compCount = Math.max(1, Math.min(k, p));
  const random = createSeededRandom(p * 37 + xMatrix.length * 11);

  for (let c = 0; c < compCount; c++) {
    let v = normalizeVector(Array.from({ length: p }, () => random() - 0.5));
    for (let it = 0; it < 50; it++) v = normalizeVector(matVec(cov, v));
    components.push(v);
    const lambda = dot(v, matVec(cov, v));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) cov[i][j] -= lambda * v[i] * v[j];
    }
  }

  return { components, means };
}

function projectToComponents(features: number[], means: number[], components: number[][]): number[] {
  const centered = features.map((value, i) => value - (means[i] ?? 0));
  return components.map((component) => dot(centered, component));
}

function rssForPowers(data: DataPoint[], powers: number[]): { rss: number; solution: LinearSolution } {
  const solution = fitLeastSquaresWithPowers(data, powers, 0);
  let rss = 0;
  for (const point of data) {
    const error = point.y - predictFromLinearSolution(point.x, powers, solution);
    rss += error * error;
  }
  return { rss, solution };
}

function calculateAic(n: number, rss: number, parameters: number): number {
  const safeRss = Math.max(rss / n, 1e-10);
  return n * Math.log(safeRss) + 2 * parameters;
}

function buildStepwiseModel(
  data: DataPoint[],
  polynomialDegree: number,
  stepwiseTerms: number,
  direction: 'forward' | 'backward'
): RegressionFit {
  const clampedDegree = Math.max(1, Math.min(polynomialDegree, 6));
  const candidates = Array.from({ length: clampedDegree }, (_, index) => index + 1);
  const targetTerms = Math.max(1, Math.min(stepwiseTerms, candidates.length));
  const n = data.length;

  if (direction === 'forward') {
    const selected: number[] = [];
    let bestAic = Infinity;

    while (selected.length < targetTerms) {
      let bestCandidate: number | null = null;
      let bestCandidateAic = Infinity;
      for (const candidate of candidates) {
        if (selected.includes(candidate)) continue;
        const currentPowers = [...selected, candidate].sort((a, b) => a - b);
        const { rss } = rssForPowers(data, currentPowers);
        const aic = calculateAic(n, rss, currentPowers.length + 1);
        if (aic < bestCandidateAic) {
          bestCandidateAic = aic;
          bestCandidate = candidate;
        }
      }

      if (bestCandidate === null) break;
      if (bestCandidateAic > bestAic && selected.length > 0) break;
      bestAic = bestCandidateAic;
      selected.push(bestCandidate);
    }

    const powers = selected.sort((a, b) => a - b);
    const solution = fitLeastSquaresWithPowers(data, powers, 0);
    return {
      modelType: 'forward_stepwise',
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: powers,
      predict: (input) => predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, powers, solution),
    };
  }

  let selected = [...candidates];
  while (selected.length > targetTerms) {
    let bestSet = selected;
    let bestAic = Infinity;

    for (const candidate of selected) {
      const trial = selected.filter((feature) => feature !== candidate);
      const { rss } = rssForPowers(data, trial);
      const aic = calculateAic(n, rss, trial.length + 1);
      if (aic < bestAic) {
        bestAic = aic;
        bestSet = trial;
      }
    }

    selected = bestSet;
  }

  const powers = selected.sort((a, b) => a - b);
  const solution = fitLeastSquaresWithPowers(data, powers, 0);
  return {
    modelType: 'backward_stepwise',
    intercept: solution.intercept,
    coefficients: solution.coefficients,
    featurePowers: powers,
    predict: (input) => predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, powers, solution),
  };
}

interface TreeSample {
  x: number[];
  y: number;
}

interface Stump {
  feature: number;
  threshold: number;
  polarity: 1 | -1;
}

interface BoostedStump {
  stump: Stump;
  alpha: number;
}

interface TreeNode {
  feature: number;
  threshold: number;
  probability: number;
  isLeaf: boolean;
  left: TreeNode | null;
  right: TreeNode | null;
}

function giniImpurity(samples: TreeSample[]): number {
  if (samples.length === 0) return 0;
  const positive = samples.reduce((sum, sample) => sum + (sample.y >= 0.5 ? 1 : 0), 0);
  const p1 = positive / samples.length;
  const p0 = 1 - p1;
  return 1 - p0 * p0 - p1 * p1;
}

function candidateThresholds(samples: TreeSample[], feature: number): number[] {
  const sorted = [...samples].sort((a, b) => a.x[feature] - b.x[feature]);
  const thresholds: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1].x[feature];
    const current = sorted[i].x[feature];
    if (current !== prev) thresholds.push((prev + current) / 2);
  }
  if (thresholds.length <= 24) return thresholds;
  const stride = Math.ceil(thresholds.length / 24);
  return thresholds.filter((_, index) => index % stride === 0);
}

function buildDecisionTree(
  samples: TreeSample[],
  depth: number,
  maxDepth: number,
  minLeafSize: number,
  featureIndices: number[] | null
): TreeNode {
  const positive = samples.reduce((sum, sample) => sum + (sample.y >= 0.5 ? 1 : 0), 0);
  const probability = positive / Math.max(samples.length, 1);

  if (
    depth >= maxDepth
    || samples.length <= minLeafSize * 2
    || probability <= 1e-6
    || probability >= 1 - 1e-6
  ) {
    return {
      feature: 0,
      threshold: 0,
      probability,
      isLeaf: true,
      left: null,
      right: null,
    };
  }

  const featureCount = samples[0]?.x.length ?? 0;
  const features = featureIndices && featureIndices.length > 0
    ? featureIndices
    : Array.from({ length: featureCount }, (_, index) => index);
  const parentImpurity = giniImpurity(samples);
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  let bestLeft: TreeSample[] = [];
  let bestRight: TreeSample[] = [];

  for (const feature of features) {
    const thresholds = candidateThresholds(samples, feature);
    for (const threshold of thresholds) {
      const left: TreeSample[] = [];
      const right: TreeSample[] = [];
      for (const sample of samples) {
        if ((sample.x[feature] ?? 0) <= threshold) left.push(sample);
        else right.push(sample);
      }
      if (left.length < minLeafSize || right.length < minLeafSize) continue;
      const score = (left.length / samples.length) * giniImpurity(left) + (right.length / samples.length) * giniImpurity(right);
      if (score < bestScore) {
        bestScore = score;
        bestFeature = feature;
        bestThreshold = threshold;
        bestLeft = left;
        bestRight = right;
      }
    }
  }

  if (bestFeature < 0 || parentImpurity - bestScore < 1e-5) {
    return {
      feature: 0,
      threshold: 0,
      probability,
      isLeaf: true,
      left: null,
      right: null,
    };
  }

  return {
    feature: bestFeature,
    threshold: bestThreshold,
    probability,
    isLeaf: false,
    left: buildDecisionTree(bestLeft, depth + 1, maxDepth, minLeafSize, featureIndices),
    right: buildDecisionTree(bestRight, depth + 1, maxDepth, minLeafSize, featureIndices),
  };
}

function predictTreeProbability(node: TreeNode, features: number[]): number {
  let current: TreeNode = node;
  while (!current.isLeaf) {
    const value = features[current.feature] ?? 0;
    current = value <= current.threshold ? (current.left ?? current) : (current.right ?? current);
    if (!current.left && !current.right && !current.isLeaf) break;
  }
  return current.probability;
}

function stumpPredict(stump: Stump, features: number[]): 0 | 1 {
  const value = features[stump.feature] ?? 0;
  const isPositive = stump.polarity === 1 ? value <= stump.threshold : value > stump.threshold;
  return isPositive ? 1 : 0;
}

function fitBestWeightedStump(samples: TreeSample[], weights: number[]): { stump: Stump; error: number } {
  const featureCount = samples[0]?.x.length ?? 1;
  let best: Stump = { feature: 0, threshold: 0, polarity: 1 };
  let bestError = Number.POSITIVE_INFINITY;

  for (let feature = 0; feature < featureCount; feature++) {
    const thresholds = candidateThresholds(samples, feature);
    if (thresholds.length === 0) thresholds.push(0);
    for (const threshold of thresholds) {
      for (const polarity of [1, -1] as const) {
        const stump: Stump = { feature, threshold, polarity };
        let error = 0;
        for (let i = 0; i < samples.length; i++) {
          const pred = stumpPredict(stump, samples[i].x);
          const truth = samples[i].y >= 0.5 ? 1 : 0;
          if (pred !== truth) error += weights[i];
        }
        if (error < bestError) {
          bestError = error;
          best = stump;
        }
      }
    }
  }
  return { stump: best, error: Math.min(Math.max(bestError, 1e-8), 1 - 1e-8) };
}

export function generateDataset(
  type: DatasetType,
  n = 50,
  seed = Date.now(),
  featureMode: '1d' | '2d' = '1d',
  recipe?: RandomDataRecipe
): DataPoint[] {
  const points: DataPoint[] = [];
  const random = createSeededRandom(seed);
  const trueSlope = 1.8;
  const trueIntercept = 0.7;

  if (isClassificationDataset(type)) {
    for (let i = 0; i < n; i++) {
      let x = -4 + random() * 8;
      let x2 = -4 + random() * 8;
      let y = 0;
      if (type === 'class_linear') {
        const margin = x + 0.8 * x2 + (random() - 0.5) * 1.35;
        y = margin > 0 ? 1 : 0;
      } else if (type === 'class_overlap') {
        const positive = random() < 0.5;
        if (positive) {
          x = 0.9 + (random() - 0.5) * 4.4;
          x2 = 0.6 + (random() - 0.5) * 4.2;
          y = 1;
        } else {
          x = -0.9 + (random() - 0.5) * 4.4;
          x2 = -0.6 + (random() - 0.5) * 4.2;
          y = 0;
        }
      } else if (type === 'class_moons') {
        const t = random() * Math.PI;
        if (i < n / 2) {
          x = Math.cos(t) * 2.2 + (random() - 0.5) * 0.45;
          x2 = Math.sin(t) * 2.2 + (random() - 0.5) * 0.45;
          y = 0;
        } else {
          x = 1 - Math.cos(t) * 2.2 + (random() - 0.5) * 0.45;
          x2 = -0.55 - Math.sin(t) * 2.2 + (random() - 0.5) * 0.45;
          y = 1;
        }
      } else {
        const positive = random() < 0.2;
        if (positive) {
          x = 1.4 + (random() - 0.5) * 2.3;
          x2 = 1.2 + (random() - 0.5) * 2.3;
          y = 1;
        } else {
          x = -1.3 + (random() - 0.5) * 3.4;
          x2 = -1.2 + (random() - 0.5) * 3.4;
          y = 0;
        }
      }
      points.push({
        x,
        x2,
        features: [x, x2],
        y: (
          random() < (type === 'class_linear'
            ? 0.03
            : type === 'class_overlap'
              ? 0.12
              : type === 'class_moons'
                ? 0.07
                : 0.05)
            ? 1 - y
            : y
        ),
      });
    }
    return points;
  }

  for (let i = 0; i < n; i++) {
    const x = -5 + (10 * i) / Math.max(n - 1, 1);
    const x2 = featureMode === '2d'
      ? (recipe?.correlatedFeatures ? x * 0.55 + (random() - 0.5) * 2.2 : -4 + random() * 8)
      : undefined;
    let y: number;
    const gaussianNoise = (random() - 0.5) * 3;
    const heavyTailNoise = ((random() - 0.5) / Math.max(0.08, random())) * 0.8;
    const heteroNoise = gaussianNoise * (0.6 + Math.abs(x) / 2.2);
    const noiseMode = type === 'random_recipe' ? (recipe?.noiseType ?? 'gaussian') : 'gaussian';
    const noise = noiseMode === 'heavy_tail' ? heavyTailNoise : noiseMode === 'heteroscedastic' ? heteroNoise : gaussianNoise;

    switch (type) {
      case 'linear':
        y = trueSlope * x + trueIntercept + (x2 !== undefined ? 1.1 * x2 : 0) + noise * 0.2;
        break;
      case 'noisy':
        y = trueSlope * x + trueIntercept + (x2 !== undefined ? 0.9 * x2 : 0) + noise * 1.2;
        break;
      case 'outliers':
        y = trueSlope * x + trueIntercept + (x2 !== undefined ? 0.75 * x2 : 0) + (random() < 0.14 ? noise * 5 : noise * 0.8);
        break;
      case 'heteroscedastic':
        y = trueSlope * x + trueIntercept + (x2 !== undefined ? 0.6 * x2 : 0) + noise * (0.7 + Math.abs(x) / 1.8);
        break;
      case 'quadratic':
        y = 0.45 * x * x + 0.9 * x - 1.2 + (x2 !== undefined ? 0.35 * x2 * x2 : 0) + noise * 0.9;
        break;
      case 'sinusoidal':
        y = 2.2 * Math.sin(x) + 0.35 * x + (x2 !== undefined ? 1.2 * Math.cos(x2) : 0) + noise;
        break;
      case 'piecewise':
        y = x < -1 ? -1.7 * x + 1.8 : x < 2 ? 0.7 * x - 0.3 : 2.4 * x - 4.2;
        y += (x2 !== undefined ? 0.5 * x2 : 0) + noise * 0.8;
        break;
      case 'random_recipe': {
        const pattern = recipe?.pattern ?? 'linear';
        if (pattern === 'linear') {
          y = trueSlope * x + trueIntercept + (x2 !== undefined ? 0.8 * x2 : 0) + noise * 0.55;
        } else if (pattern === 'polynomial') {
          y = 0.35 * x * x + 0.75 * x - 0.7 + (x2 !== undefined ? 0.22 * x2 * x2 : 0) + noise * 0.65;
        } else if (pattern === 'sinusoidal') {
          y = 2.0 * Math.sin(x) + 0.25 * x + (x2 !== undefined ? 0.8 * Math.cos(x2) : 0) + noise * 0.7;
        } else if (pattern === 'piecewise') {
          y = x < -1.5 ? -1.5 * x + 1.4 : x < 1.5 ? 0.6 * x - 0.4 : 2.0 * x - 2.5;
          y += (x2 !== undefined ? 0.45 * x2 : 0) + noise * 0.6;
        } else {
          y = 0.45 * x * x + 1.3 * Math.sin(x) + 0.45 * x + (x2 !== undefined ? 0.35 * x2 : 0) + noise * 0.8;
        }
        const outlierLevel = recipe?.outlierLevel ?? 'none';
        const outlierProb = outlierLevel === 'none' ? 0 : outlierLevel === 'low' ? 0.03 : outlierLevel === 'medium' ? 0.08 : 0.14;
        if (random() < outlierProb) y += noise * 6.5;
        break;
      }
      default:
        y = trueSlope * x + trueIntercept + (x2 !== undefined ? x2 : 0) + noise * 0.2;
    }

    points.push({ x, x2, features: x2 !== undefined ? [x, x2] : [x], y });
  }

  return points;
}

export function fitRegressionModel(data: DataPoint[], modelType: ModelType, params: ModelParams): RegressionFit {
  if (data.length === 0) {
    return {
      modelType,
      intercept: 0,
      coefficients: [],
      featurePowers: [],
      predict: () => 0,
    };
  }
  const featureCount = getPointFeatures(data[0]).length;

  if (modelType === 'logistic_classifier') {
    const p = featureCount;
    const weights = Array.from({ length: p }, () => 0);
    let bias = 0;
    const lr = 0.08;
    const lambda = Math.max(params.alpha, 0) * 0.01;
    for (let iteration = 0; iteration < 500; iteration++) {
      const grad = Array.from({ length: p }, () => 0);
      let gradBias = 0;
      for (const point of data) {
        const x = getPointFeatures(point);
        let linear = bias;
        for (let j = 0; j < p; j++) linear += weights[j] * x[j];
        const pred = sigmoid(linear);
        const error = pred - point.y;
        for (let j = 0; j < p; j++) grad[j] += error * x[j];
        gradBias += error;
      }
      for (let j = 0; j < p; j++) {
        weights[j] -= lr * (grad[j] / data.length + lambda * weights[j]);
      }
      bias -= lr * (gradBias / data.length);
    }
    return {
      modelType,
      intercept: bias,
      coefficients: weights,
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        let linear = bias;
        for (let j = 0; j < p; j++) linear += weights[j] * (x[j] ?? 0);
        return sigmoid(linear);
      },
    };
  }

  if (modelType === 'knn_classifier') {
    const trainPoints = data.map((point) => ({ x: getPointFeatures(point), y: point.y }));
    const k = Math.max(1, Math.round(params.knnK));
    return {
      modelType,
      intercept: 0,
      coefficients: [],
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        const neighbors = trainPoints
          .map((point) => {
            const distance = point.x.reduce((sum, value, idx) => sum + Math.pow(value - (x[idx] ?? 0), 2), 0);
            return { distance, y: point.y };
          })
          .sort((a, b) => a.distance - b.distance)
          .slice(0, Math.min(k, trainPoints.length));
        const positive = neighbors.reduce((sum, neighbor) => sum + (neighbor.y >= 0.5 ? 1 : 0), 0);
        return positive / Math.max(neighbors.length, 1);
      },
    };
  }

  if (modelType === 'svm_classifier') {
    const gamma = Math.max(params.svmGamma, 0.05);
    const c = Math.max(params.svmC, 0.01);
    const centers = seededShuffle(
      data.map((point) => getPointFeatures(point)),
      data.length * 17 + 13
    ).slice(0, Math.min(30, data.length));
    const weights = Array.from({ length: centers.length }, () => 0);
    let bias = 0;
    const lr = 0.08;
    const lambda = 1 / (c * Math.max(data.length, 1));

    for (let iteration = 0; iteration < 450; iteration++) {
      const grad = Array.from({ length: centers.length }, () => 0);
      let gradBias = 0;
      for (const point of data) {
        const x = getPointFeatures(point);
        const phi = centers.map((center) => rbfKernel(x, center, gamma));
        let score = bias;
        for (let j = 0; j < weights.length; j++) score += weights[j] * phi[j];
        const pred = sigmoid(score);
        const error = pred - point.y;
        for (let j = 0; j < weights.length; j++) grad[j] += error * phi[j];
        gradBias += error;
      }
      for (let j = 0; j < weights.length; j++) {
        weights[j] -= lr * (grad[j] / data.length + lambda * weights[j]);
      }
      bias -= lr * (gradBias / data.length);
    }

    return {
      modelType,
      intercept: bias,
      coefficients: weights,
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        let score = bias;
        for (let j = 0; j < weights.length; j++) {
          score += weights[j] * rbfKernel(x, centers[j], gamma);
        }
        return sigmoid(score);
      },
    };
  }

  if (modelType === 'decision_tree_classifier') {
    const trainSamples = data.map((point) => ({ x: getPointFeatures(point), y: point.y }));
    const maxDepth = Math.max(1, Math.min(10, Math.round(params.treeDepth)));
    const tree = buildDecisionTree(trainSamples, 0, maxDepth, 3, null);
    return {
      modelType,
      intercept: 0,
      coefficients: [],
      featurePowers: [],
      predict: (input) => predictTreeProbability(tree, toFeatureArray(input)),
    };
  }

  if (modelType === 'random_forest_classifier') {
    const trainSamples = data.map((point) => ({ x: getPointFeatures(point), y: point.y }));
    const treeCount = Math.max(3, Math.min(150, Math.round(params.forestTrees)));
    const maxDepth = Math.max(1, Math.min(10, Math.round(params.treeDepth)));
    const featureCountLocal = trainSamples[0]?.x.length ?? 1;
    const mtry = Math.max(1, Math.floor(Math.sqrt(featureCountLocal)));
    const seedBase = Math.round(
      trainSamples.reduce((sum, sample, idx) => sum + (idx + 1) * ((sample.x[0] ?? 0) + (sample.x[1] ?? 0) + sample.y), 0) * 1000
    );
    const random = createSeededRandom(Math.abs(seedBase) + 97);
    const trees: TreeNode[] = [];

    for (let treeIndex = 0; treeIndex < treeCount; treeIndex++) {
      const bootstrap: TreeSample[] = [];
      for (let i = 0; i < trainSamples.length; i++) {
        bootstrap.push(trainSamples[Math.floor(random() * trainSamples.length)]);
      }
      const featurePool = Array.from({ length: featureCountLocal }, (_, idx) => idx);
      const selectedFeatures = seededShuffle(featurePool, Math.floor(random() * 1_000_000)).slice(0, mtry);
      trees.push(buildDecisionTree(bootstrap, 0, maxDepth, 3, selectedFeatures));
    }

    return {
      modelType,
      intercept: 0,
      coefficients: [],
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        const prob = trees.reduce((sum, tree) => sum + predictTreeProbability(tree, x), 0) / Math.max(trees.length, 1);
        return Math.min(1 - 1e-6, Math.max(1e-6, prob));
      },
    };
  }

  if (modelType === 'adaboost_classifier') {
    const samples = data.map((point) => ({ x: getPointFeatures(point), y: point.y }));
    const rounds = Math.max(5, Math.min(200, Math.round(params.boostingRounds)));
    const learningRate = Math.max(0.01, Math.min(1, params.learningRate));
    let weights = Array.from({ length: samples.length }, () => 1 / samples.length);
    const learners: BoostedStump[] = [];

    for (let t = 0; t < rounds; t++) {
      const { stump, error } = fitBestWeightedStump(samples, weights);
      const alpha = 0.5 * Math.log((1 - error) / error) * learningRate;
      learners.push({ stump, alpha });

      let weightSum = 0;
      for (let i = 0; i < samples.length; i++) {
        const truth = samples[i].y >= 0.5 ? 1 : -1;
        const pred = stumpPredict(stump, samples[i].x) === 1 ? 1 : -1;
        weights[i] *= Math.exp(-alpha * truth * pred);
        weightSum += weights[i];
      }
      if (weightSum <= 0) break;
      weights = weights.map((w) => w / weightSum);
    }

    return {
      modelType,
      intercept: 0,
      coefficients: [],
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        const score = learners.reduce((sum, learner) => {
          const pred = stumpPredict(learner.stump, x) === 1 ? 1 : -1;
          return sum + learner.alpha * pred;
        }, 0);
        return sigmoid(2 * score);
      },
    };
  }

  if (modelType === 'gradient_boosting_classifier') {
    const samples = data.map((point) => ({ x: getPointFeatures(point), y: point.y }));
    const rounds = Math.max(5, Math.min(200, Math.round(params.boostingRounds)));
    const learningRate = Math.max(0.01, Math.min(1, params.learningRate));
    const stumps: Array<{ stump: Stump; weight: number }> = [];
    let bias = 0;

    for (let t = 0; t < rounds; t++) {
      const probs = samples.map((sample) => sigmoid(bias + stumps.reduce((sum, s) => sum + s.weight * (stumpPredict(s.stump, sample.x) ? 1 : -1), 0)));
      const pseudoResidual = samples.map((sample, i) => sample.y - probs[i]);
      const sampleWeights = pseudoResidual.map((value) => Math.abs(value) + 1e-4);
      const weightNorm = sampleWeights.reduce((sum, value) => sum + value, 0);
      const normalized = sampleWeights.map((value) => value / Math.max(weightNorm, 1e-8));
      const { stump } = fitBestWeightedStump(samples, normalized);
      let gammaNumerator = 0;
      let gammaDenominator = 0;
      for (let i = 0; i < samples.length; i++) {
        const direction = stumpPredict(stump, samples[i].x) === 1 ? 1 : -1;
        gammaNumerator += pseudoResidual[i] * direction;
        gammaDenominator += direction * direction;
      }
      const gamma = learningRate * (gammaNumerator / Math.max(gammaDenominator, 1e-8));
      stumps.push({ stump, weight: gamma });
      bias += learningRate * mean(pseudoResidual) * 0.25;
    }

    return {
      modelType,
      intercept: bias,
      coefficients: [],
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        const raw = bias + stumps.reduce((sum, stump) => sum + stump.weight * (stumpPredict(stump.stump, x) === 1 ? 1 : -1), 0);
        return sigmoid(raw);
      },
    };
  }

  if (modelType === 'svm_regressor') {
    const gamma = Math.max(params.svmGamma ?? 1, 0.01);
    const c = Math.max(params.svmC ?? 1, 0.01);
    const centers = seededShuffle(
      data.map((point) => getPointFeatures(point)),
      data.length * 41 + 17
    ).slice(0, Math.min(40, data.length));
    const phi = data.map((point) => centers.map((center) => rbfKernel(getPointFeatures(point), center, gamma)));
    const p = centers.length;
    const xtx = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
    const xty = Array.from({ length: p }, () => 0);
    const y = data.map((point) => point.y);
    const yMean = mean(y);
    const phiMeans = Array.from({ length: p }, (_, j) => mean(phi.map((row) => row[j])));
    for (let i = 0; i < phi.length; i++) {
      for (let a = 0; a < p; a++) {
        const va = phi[i][a] - phiMeans[a];
        xty[a] += va * (y[i] - yMean);
        for (let b = 0; b < p; b++) xtx[a][b] += va * (phi[i][b] - phiMeans[b]);
      }
    }
    const ridge = 1 / c;
    for (let i = 0; i < p; i++) xtx[i][i] += ridge * data.length;
    const weights = solveLinearSystem(xtx, xty);
    const intercept = yMean - weights.reduce((sum, w, i) => sum + w * phiMeans[i], 0);
    return {
      modelType,
      intercept,
      coefficients: weights,
      featurePowers: [],
      predict: (input) => {
        const x = toFeatureArray(input);
        let pred = intercept;
        for (let j = 0; j < weights.length; j++) pred += weights[j] * rbfKernel(x, centers[j], gamma);
        return pred;
      },
    };
  }

  if (modelType === 'pcr_regressor' || modelType === 'pls_regressor') {
    const xMatrix = data.map((point) => getPointFeatures(point));
    const y = data.map((point) => point.y);
    const compCount = Math.max(
      1,
      Math.min(modelType === 'pls_regressor' ? (params.plsComponents ?? 2) : (params.pcaComponents ?? 2), xMatrix[0]?.length ?? 1)
    );

    if (modelType === 'pcr_regressor') {
      const { components, means } = computePcaComponents(xMatrix, compCount);
      const scores = xMatrix.map((features) => projectToComponents(features, means, components));
      const scoreData = scores.map((s, i) => ({ x: s[0] ?? 0, x2: s[1], features: s, y: y[i] }));
      const solution = fitLeastSquaresMulti(scoreData, Math.max(params.alpha, 0));
      return {
        modelType,
        intercept: solution.intercept,
        coefficients: solution.coefficients,
        featurePowers: [],
        predict: (input) => {
          const features = toFeatureArray(input);
          const proj = projectToComponents(features, means, components);
          return predictFromMultiSolution(proj, solution);
        },
      };
    }

    // Simplified PLS via iterative latent directions using covariance with y.
    const p = xMatrix[0]?.length ?? 1;
    const meansX = Array.from({ length: p }, (_, j) => mean(xMatrix.map((row) => row[j])));
    const centeredX = xMatrix.map((row) => row.map((value, j) => value - meansX[j]));
    const yMean = mean(y);
    const centeredY = y.map((value) => value - yMean);
    const components: number[][] = [];
    let Xwork = centeredX.map((row) => [...row]);
    let ywork = [...centeredY];
    for (let c = 0; c < compCount; c++) {
      const wRaw = Array.from({ length: p }, (_, j) =>
        Xwork.reduce((sum, row, i) => sum + row[j] * ywork[i], 0)
      );
      const w = normalizeVector(wRaw);
      components.push(w);
      const t = Xwork.map((row) => dot(row, w));
      const denom = Math.max(dot(t, t), 1e-8);
      const pLoad = Array.from({ length: p }, (_, j) => Xwork.reduce((sum, row, i) => sum + row[j] * t[i], 0) / denom);
      const q = dot(ywork, t) / denom;
      Xwork = Xwork.map((row, i) => row.map((value, j) => value - t[i] * pLoad[j]));
      ywork = ywork.map((value, i) => value - q * t[i]);
    }
    const scores = centeredX.map((row) => components.map((comp) => dot(row, comp)));
    const scoreData = scores.map((s, i) => ({ x: s[0] ?? 0, x2: s[1], features: s, y: y[i] }));
    const solution = fitLeastSquaresMulti(scoreData, Math.max(params.alpha, 0));
    return {
      modelType,
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: [],
      predict: (input) => {
        const features = toFeatureArray(input).map((value, j) => value - (meansX[j] ?? 0));
        const proj = components.map((comp) => dot(features, comp));
        return predictFromMultiSolution(proj, solution);
      },
    };
  }

  if (modelType === 'ols') {
    const powers = featureCount === 1 ? [1] : [];
    const solution = featureCount === 1 ? fitLeastSquaresWithPowers(data, powers, 0) : fitLeastSquaresMulti(data, 0);
    return {
      modelType,
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: powers,
      predict: (input) =>
        featureCount === 1
          ? predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, [1], solution)
          : predictFromMultiSolution(input, solution),
    };
  }

  if (modelType === 'ridge') {
    const powers = featureCount === 1 ? [1] : [];
    const solution = featureCount === 1
      ? fitLeastSquaresWithPowers(data, powers, Math.max(params.alpha, 0))
      : fitLeastSquaresMulti(data, Math.max(params.alpha, 0));
    return {
      modelType,
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: powers,
      predict: (input) =>
        featureCount === 1
          ? predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, [1], solution)
          : predictFromMultiSolution(input, solution),
    };
  }

  if (modelType === 'lasso') {
    const powers = featureCount === 1 ? [1] : [];
    const solution = featureCount === 1
      ? fitCoordinateDescent(data, powers, Math.max(params.alpha, 0), 1)
      : fitCoordinateDescentMulti(data, Math.max(params.alpha, 0), 1);
    return {
      modelType,
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: powers,
      predict: (input) =>
        featureCount === 1
          ? predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, [1], solution)
          : predictFromMultiSolution(input, solution),
    };
  }

  if (modelType === 'elasticnet') {
    const powers = featureCount === 1 ? [1] : [];
    const solution = featureCount === 1
      ? fitCoordinateDescent(
        data,
        powers,
        Math.max(params.alpha, 0),
        Math.min(Math.max(params.l1Ratio, 0), 1)
      )
      : fitCoordinateDescentMulti(data, Math.max(params.alpha, 0), Math.min(Math.max(params.l1Ratio, 0), 1));
    return {
      modelType,
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: powers,
      predict: (input) =>
        featureCount === 1
          ? predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, [1], solution)
          : predictFromMultiSolution(input, solution),
    };
  }

  if (modelType === 'polynomial') {
    if (featureCount > 1) {
      const degree = Math.max(1, Math.min(params.polynomialDegree, 6));
      const transformed = data.map((point) => {
        const f = getPointFeatures(point);
        return {
          ...point,
          features: expandPolynomialFeatures(f, degree),
        };
      });
      const solution = fitLeastSquaresMulti(transformed, 0);
      return {
        modelType,
        intercept: solution.intercept,
        coefficients: solution.coefficients,
        featurePowers: [],
        predict: (input) => predictFromMultiSolution(expandPolynomialFeatures(toFeatureArray(input), degree), solution),
      };
    }
    const degree = Math.max(1, Math.min(params.polynomialDegree, 6));
    const powers = Array.from({ length: degree }, (_, index) => index + 1);
    const solution = fitLeastSquaresWithPowers(data, powers, 0);
    return {
      modelType,
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: powers,
      predict: (input) => predictFromLinearSolution(typeof input === 'number' ? input : input[0] ?? 0, powers, solution),
    };
  }

  if (modelType === 'forward_stepwise') {
    if (featureCount > 1) {
      const solution = fitLeastSquaresMulti(data, 0);
      return {
        modelType: 'ols',
        intercept: solution.intercept,
        coefficients: solution.coefficients,
        featurePowers: [],
        predict: (input) => predictFromMultiSolution(input, solution),
      };
    }
    return buildStepwiseModel(data, params.polynomialDegree, params.stepwiseTerms, 'forward');
  }
  if (featureCount > 1) {
    const solution = fitLeastSquaresMulti(data, 0);
    return {
      modelType: 'ols',
      intercept: solution.intercept,
      coefficients: solution.coefficients,
      featurePowers: [],
      predict: (input) => predictFromMultiSolution(input, solution),
    };
  }
  return buildStepwiseModel(data, params.polynomialDegree, params.stepwiseTerms, 'backward');
}

export function getPredictions(data: DataPoint[], fit: RegressionFit): number[] {
  return data.map((point) => fit.predict(getPointFeatures(point)));
}

export function splitDataset(
  data: DataPoint[],
  testRatio: number,
  seed: number,
  stratifyByClass = false
): { train: DataPoint[]; test: DataPoint[] } {
  if (data.length < 3) {
    return { train: data, test: [] };
  }
  const clampedRatio = Math.min(Math.max(testRatio, 0.1), 0.45);
  const testIndices = new Set<number>();

  if (stratifyByClass) {
    const positiveIndices = data.map((point, index) => ({ point, index })).filter(({ point }) => point.y >= 0.5).map(({ index }) => index);
    const negativeIndices = data.map((point, index) => ({ point, index })).filter(({ point }) => point.y < 0.5).map(({ index }) => index);
    const shuffledPos = seededShuffle(positiveIndices, seed + 101);
    const shuffledNeg = seededShuffle(negativeIndices, seed + 203);
    const posTestCount = Math.max(1, Math.floor(shuffledPos.length * clampedRatio));
    const negTestCount = Math.max(1, Math.floor(shuffledNeg.length * clampedRatio));
    for (const idx of shuffledPos.slice(0, posTestCount)) testIndices.add(idx);
    for (const idx of shuffledNeg.slice(0, negTestCount)) testIndices.add(idx);
  } else {
    const indices = seededShuffle(
      data.map((_, index) => index),
      seed
    );
    const testCount = Math.max(1, Math.floor(data.length * clampedRatio));
    for (const idx of indices.slice(0, testCount)) testIndices.add(idx);
  }

  const train: DataPoint[] = [];
  const test: DataPoint[] = [];
  for (let index = 0; index < data.length; index++) {
    if (testIndices.has(index)) {
      test.push(data[index]);
    } else {
      train.push(data[index]);
    }
  }
  return { train, test };
}

export function computeMetricsFromPredictions(
  yTrue: number[],
  yPred: number[],
  featureCount = 1
): { r2: number; rmse: number; mae: number; mse: number; mape: number; explainedVariance: number; medianAe: number; adjustedR2: number; accuracy: number; precision: number; recall: number; specificity: number; f1: number; rocAuc: number; prAuc: number; logLoss: number } {
  const n = yTrue.length;
  if (n === 0) return { r2: 0, rmse: 0, mae: 0, mse: 0, mape: 0, explainedVariance: 0, medianAe: 0, adjustedR2: 0, accuracy: 0, precision: 0, recall: 0, specificity: 0, f1: 0, rocAuc: 0, prAuc: 0, logLoss: 0 };

  const yMean = mean(yTrue);
  let ssTotal = 0;
  let ssResidual = 0;
  let mae = 0;
  let mape = 0;
  const absoluteErrors: number[] = [];
  const residuals: number[] = [];
  for (let i = 0; i < n; i++) {
    const residual = yTrue[i] - yPred[i];
    ssResidual += residual * residual;
    ssTotal += Math.pow(yTrue[i] - yMean, 2);
    mae += Math.abs(residual);
    absoluteErrors.push(Math.abs(residual));
    residuals.push(residual);
    const denom = Math.max(Math.abs(yTrue[i]), 1e-8);
    mape += Math.abs(residual) / denom;
  }

  const mse = ssResidual / n;
  const residualMean = mean(residuals);
  const residualVariance = mean(residuals.map((value) => Math.pow(value - residualMean, 2)));
  const yVariance = mean(yTrue.map((value) => Math.pow(value - yMean, 2)));
  const r2 = ssTotal === 0 ? 0 : 1 - ssResidual / ssTotal;
  const adjustedR2 = n > featureCount + 1 ? 1 - (1 - r2) * ((n - 1) / (n - featureCount - 1)) : r2;
  return {
    r2,
    rmse: Math.sqrt(mse),
    mae: mae / n,
    mse,
    mape: (mape / n) * 100,
    explainedVariance: yVariance === 0 ? 0 : 1 - residualVariance / yVariance,
    medianAe: median(absoluteErrors),
    adjustedR2,
    accuracy: 0,
    precision: 0,
    recall: 0,
    specificity: 0,
    f1: 0,
    rocAuc: 0,
    prAuc: 0,
    logLoss: 0,
  };
}

function rocAucScore(yTrue: number[], yProb: number[]): number {
  const positives = yTrue.filter((v) => v >= 0.5).length;
  const negatives = yTrue.length - positives;
  if (positives === 0 || negatives === 0) return 0.5;
  let wins = 0;
  let ties = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] < 0.5) continue;
    for (let j = 0; j < yTrue.length; j++) {
      if (yTrue[j] >= 0.5) continue;
      if (yProb[i] > yProb[j]) wins += 1;
      else if (Math.abs(yProb[i] - yProb[j]) < 1e-12) ties += 1;
    }
  }
  return (wins + 0.5 * ties) / (positives * negatives);
}

function prAucScore(yTrue: number[], yProb: number[]): number {
  const points = yTrue.map((label, index) => ({ label, score: yProb[index] })).sort((a, b) => b.score - a.score);
  const totalPositives = yTrue.filter((v) => v >= 0.5).length;
  if (totalPositives === 0) return 0;
  let tp = 0;
  let fp = 0;
  const curve: Array<{ recall: number; precision: number }> = [{ recall: 0, precision: 1 }];
  for (const point of points) {
    if (point.label >= 0.5) tp += 1;
    else fp += 1;
    const precision = tp / Math.max(tp + fp, 1);
    const recall = tp / totalPositives;
    curve.push({ recall, precision });
  }
  let area = 0;
  for (let i = 1; i < curve.length; i++) {
    const x1 = curve[i - 1].recall;
    const x2 = curve[i].recall;
    const y = curve[i].precision;
    area += (x2 - x1) * y;
  }
  return area;
}

function classificationMetrics(
  yTrue: number[],
  yProb: number[],
  threshold: number
): { r2: number; rmse: number; mae: number; mse: number; mape: number; explainedVariance: number; medianAe: number; adjustedR2: number; accuracy: number; precision: number; recall: number; specificity: number; f1: number; rocAuc: number; prAuc: number; logLoss: number } {
  const clipped = yProb.map((value) => Math.min(1 - 1e-6, Math.max(1e-6, value)));
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let loss = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const truth = yTrue[i] >= 0.5 ? 1 : 0;
    const pred = clipped[i] >= threshold ? 1 : 0;
    if (truth === 1 && pred === 1) tp += 1;
    if (truth === 0 && pred === 0) tn += 1;
    if (truth === 0 && pred === 1) fp += 1;
    if (truth === 1 && pred === 0) fn += 1;
    loss += -(truth * Math.log(clipped[i]) + (1 - truth) * Math.log(1 - clipped[i]));
  }
  const accuracy = (tp + tn) / Math.max(yTrue.length, 1);
  const precision = tp / Math.max(tp + fp, 1);
  const recall = tp / Math.max(tp + fn, 1);
  const specificity = tn / Math.max(tn + fp, 1);
  const f1 = (2 * precision * recall) / Math.max(precision + recall, 1e-8);
  return {
    r2: 0,
    rmse: 0,
    mae: 0,
    mse: 0,
    mape: 0,
    explainedVariance: 0,
    medianAe: 0,
    adjustedR2: 0,
    accuracy,
    precision,
    recall,
    specificity,
    f1,
    rocAuc: rocAucScore(yTrue, clipped),
    prAuc: prAucScore(yTrue, clipped),
    logLoss: loss / Math.max(yTrue.length, 1),
  };
}

export interface ClassificationDiagnostics {
  confusion: [[number, number], [number, number]];
  rocCurve: Array<{ fpr: number; tpr: number }>;
  prCurve: Array<{ recall: number; precision: number }>;
  rocAuc: number;
  prAuc: number;
}

export function computeClassificationDiagnostics(
  yTrue: number[],
  yProb: number[],
  threshold: number
): ClassificationDiagnostics {
  const clipped = yProb.map((value) => Math.min(1 - 1e-6, Math.max(1e-6, value)));
  const points = yTrue.map((label, index) => ({ label: label >= 0.5 ? 1 : 0, score: clipped[index] })).sort((a, b) => b.score - a.score);
  let tp = 0;
  let fp = 0;
  const totalPos = points.filter((point) => point.label === 1).length;
  const totalNeg = Math.max(points.length - totalPos, 1);
  const rocCurve: Array<{ fpr: number; tpr: number }> = [{ fpr: 0, tpr: 0 }];
  const prCurve: Array<{ recall: number; precision: number }> = [{ recall: 0, precision: 1 }];
  for (const point of points) {
    if (point.label === 1) tp += 1;
    else fp += 1;
    const tpr = tp / Math.max(totalPos, 1);
    const fpr = fp / totalNeg;
    const precision = tp / Math.max(tp + fp, 1);
    rocCurve.push({ fpr, tpr });
    prCurve.push({ recall: tpr, precision });
  }
  rocCurve.push({ fpr: 1, tpr: 1 });
  const predicted = clipped.map((prob) => (prob >= threshold ? 1 : 0));
  let tn = 0;
  let tpAt = 0;
  let fpAt = 0;
  let fn = 0;
  for (let i = 0; i < predicted.length; i++) {
    const truth = yTrue[i] >= 0.5 ? 1 : 0;
    const pred = predicted[i];
    if (truth === 1 && pred === 1) tpAt += 1;
    if (truth === 0 && pred === 0) tn += 1;
    if (truth === 0 && pred === 1) fpAt += 1;
    if (truth === 1 && pred === 0) fn += 1;
  }
  return {
    confusion: [[tn, fpAt], [fn, tpAt]],
    rocCurve,
    prCurve,
    rocAuc: rocAucScore(yTrue, clipped),
    prAuc: prAucScore(yTrue, clipped),
  };
}

export function computeMetrics(data: DataPoint[], fit: RegressionFit): { r2: number; rmse: number; mae: number; mse: number; mape: number; explainedVariance: number; medianAe: number; adjustedR2: number; accuracy: number; precision: number; recall: number; specificity: number; f1: number; rocAuc: number; prAuc: number; logLoss: number } {
  const yTrue = data.map((point) => point.y);
  const yPred = getPredictions(data, fit);
  const featureCount = data.length > 0 ? getPointFeatures(data[0]).length : 1;
  return computeMetricsFromPredictions(yTrue, yPred, featureCount);
}

export function evaluateModelMetrics(
  data: DataPoint[],
  modelType: ModelType,
  params: ModelParams,
  evaluationMode: EvaluationMode,
  testRatio: number,
  cvFolds: number,
  seed: number
): { r2: number; rmse: number; mae: number; mse: number; mape: number; explainedVariance: number; medianAe: number; adjustedR2: number; accuracy: number; precision: number; recall: number; specificity: number; f1: number; rocAuc: number; prAuc: number; logLoss: number } {
  if (data.length === 0) return { r2: 0, rmse: 0, mae: 0, mse: 0, mape: 0, explainedVariance: 0, medianAe: 0, adjustedR2: 0, accuracy: 0, precision: 0, recall: 0, specificity: 0, f1: 0, rocAuc: 0, prAuc: 0, logLoss: 0 };
  const featureCount = getPointFeatures(data[0]).length;
  const threshold = Math.min(0.95, Math.max(0.05, params.decisionThreshold));
  const classification = isClassificationModel(modelType);

  if (evaluationMode === 'full') {
    const fit = fitRegressionModel(data, modelType, params);
    if (!classification) return computeMetrics(data, fit);
    const yTrue = data.map((point) => point.y);
    const yProb = data.map((point) => fit.predict(getPointFeatures(point)));
    return classificationMetrics(yTrue, yProb, threshold);
  }

  if (evaluationMode === 'train_test') {
    const { train, test } = splitDataset(data, testRatio, seed, classification);
    if (train.length < 2 || test.length === 0) {
      const fit = fitRegressionModel(data, modelType, params);
      if (classification) {
        const yTrue = data.map((point) => point.y);
        const yProb = data.map((point) => fit.predict(getPointFeatures(point)));
        return classificationMetrics(yTrue, yProb, threshold);
      }
      return computeMetrics(data, fit);
    }
    const fit = fitRegressionModel(train, modelType, params);
    const yTrue = test.map((point) => point.y);
    const yPred = test.map((point) => fit.predict(getPointFeatures(point)));
    return classification ? classificationMetrics(yTrue, yPred, threshold) : computeMetricsFromPredictions(yTrue, yPred, featureCount);
  }

  const folds = Math.min(Math.max(cvFolds, 3), Math.min(10, data.length - 1));
  const shuffled = seededShuffle(
    data.map((_, index) => index),
    seed + 31
  );
  const foldSize = Math.floor(shuffled.length / folds);
  const foldMetrics: Array<{ r2: number; rmse: number; mae: number; mse: number; mape: number; explainedVariance: number; medianAe: number; adjustedR2: number; accuracy: number; precision: number; recall: number; specificity: number; f1: number; rocAuc: number; prAuc: number; logLoss: number }> = [];

  for (let fold = 0; fold < folds; fold++) {
    const start = fold * foldSize;
    const end = fold === folds - 1 ? shuffled.length : (fold + 1) * foldSize;
    const testSet = new Set(shuffled.slice(start, end));
    const train: DataPoint[] = [];
    const test: DataPoint[] = [];

    for (let index = 0; index < data.length; index++) {
      if (testSet.has(index)) {
        test.push(data[index]);
      } else {
        train.push(data[index]);
      }
    }

    if (train.length < 2 || test.length === 0) continue;
    const fit = fitRegressionModel(train, modelType, params);
    const yTrue = test.map((point) => point.y);
    const yPred = test.map((point) => fit.predict(getPointFeatures(point)));
    foldMetrics.push(classification ? classificationMetrics(yTrue, yPred, threshold) : computeMetricsFromPredictions(yTrue, yPred, featureCount));
  }

  if (foldMetrics.length === 0) {
    const fit = fitRegressionModel(data, modelType, params);
    if (classification) {
      const yTrue = data.map((point) => point.y);
      const yProb = data.map((point) => fit.predict(getPointFeatures(point)));
      return classificationMetrics(yTrue, yProb, threshold);
    }
    return computeMetrics(data, fit);
  }

  return {
    r2: mean(foldMetrics.map((metric) => metric.r2)),
    rmse: mean(foldMetrics.map((metric) => metric.rmse)),
    mae: mean(foldMetrics.map((metric) => metric.mae)),
    mse: mean(foldMetrics.map((metric) => metric.mse)),
    mape: mean(foldMetrics.map((metric) => metric.mape)),
    explainedVariance: mean(foldMetrics.map((metric) => metric.explainedVariance)),
    medianAe: mean(foldMetrics.map((metric) => metric.medianAe)),
    adjustedR2: mean(foldMetrics.map((metric) => metric.adjustedR2)),
    accuracy: mean(foldMetrics.map((metric) => metric.accuracy)),
    precision: mean(foldMetrics.map((metric) => metric.precision)),
    recall: mean(foldMetrics.map((metric) => metric.recall)),
    specificity: mean(foldMetrics.map((metric) => metric.specificity)),
    f1: mean(foldMetrics.map((metric) => metric.f1)),
    rocAuc: mean(foldMetrics.map((metric) => metric.rocAuc)),
    prAuc: mean(foldMetrics.map((metric) => metric.prAuc)),
    logLoss: mean(foldMetrics.map((metric) => metric.logLoss)),
  };
}

export function computeClassificationComplexityCurve(
  data: DataPoint[],
  modelType: ModelType,
  params: ModelParams,
  seed: number
): { complexity: number[]; trainLogLoss: number[]; validationLogLoss: number[]; trainF1: number[]; validationF1: number[]; label: string } {
  if (data.length < 20 || !isClassificationModel(modelType)) {
    return { complexity: [], trainLogLoss: [], validationLogLoss: [], trainF1: [], validationF1: [], label: 'Complexity' };
  }
  const { train, test } = splitDataset(data, 0.3, seed + 17, true);
  if (train.length < 10 || test.length < 6) {
    return { complexity: [], trainLogLoss: [], validationLogLoss: [], trainF1: [], validationF1: [], label: 'Complexity' };
  }
  let sweep: number[] = [];
  let label = 'Complexity';
  if (modelType === 'logistic_classifier') {
    sweep = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2];
    label = 'Regularization (alpha)';
  } else if (modelType === 'knn_classifier') {
    sweep = [1, 3, 5, 7, 11, 15, 21];
    label = 'Neighbors (k)';
  } else if (modelType === 'svm_classifier') {
    sweep = [0.3, 0.6, 1, 1.5, 2.5, 3.5];
    label = 'SVM C';
  } else if (modelType === 'decision_tree_classifier') {
    sweep = [1, 2, 3, 4, 5, 6, 8];
    label = 'Tree depth';
  } else if (modelType === 'random_forest_classifier') {
    sweep = [5, 10, 20, 35, 50, 75, 100];
    label = 'Number of trees';
  } else if (modelType === 'adaboost_classifier' || modelType === 'gradient_boosting_classifier') {
    sweep = [10, 20, 35, 50, 75, 100, 150];
    label = 'Boosting rounds';
  }

  const trainLogLoss: number[] = [];
  const validationLogLoss: number[] = [];
  const trainF1: number[] = [];
  const validationF1: number[] = [];

  for (const value of sweep) {
    const tuned: ModelParams = { ...params };
    if (modelType === 'logistic_classifier') tuned.alpha = value;
    if (modelType === 'knn_classifier') tuned.knnK = Math.round(value);
    if (modelType === 'svm_classifier') tuned.svmC = value;
    if (modelType === 'decision_tree_classifier') tuned.treeDepth = Math.round(value);
    if (modelType === 'random_forest_classifier') tuned.forestTrees = Math.round(value);
    if (modelType === 'adaboost_classifier' || modelType === 'gradient_boosting_classifier') tuned.boostingRounds = Math.round(value);

    const trainFit = fitRegressionModel(train, modelType, tuned);
    const trainProb = train.map((point) => trainFit.predict(getPointFeatures(point)));
    const trainTruth = train.map((point) => point.y);
    const trainMetrics = classificationMetrics(trainTruth, trainProb, tuned.decisionThreshold);
    trainLogLoss.push(trainMetrics.logLoss);
    trainF1.push(trainMetrics.f1);

    const valProb = test.map((point) => trainFit.predict(getPointFeatures(point)));
    const valTruth = test.map((point) => point.y);
    const valMetrics = classificationMetrics(valTruth, valProb, tuned.decisionThreshold);
    validationLogLoss.push(valMetrics.logLoss);
    validationF1.push(valMetrics.f1);
  }

  return { complexity: sweep, trainLogLoss, validationLogLoss, trainF1, validationF1, label };
}

export function computeBiasVarianceCurve(
  data: DataPoint[],
  seed: number,
  featureMode: '1d' | '2d'
): { complexity: number[]; trainMse: number[]; validationMse: number[] } {
  if (data.length < 8) {
    return { complexity: [], trainMse: [], validationMse: [] };
  }

  const { train, test } = splitDataset(data, 0.3, seed + 7);
  if (train.length < 5 || test.length < 3) {
    return { complexity: [], trainMse: [], validationMse: [] };
  }

  if (featureMode === '1d') {
    const complexity = [1, 2, 3, 4, 5, 6];
    const trainMse = complexity.map((degree) => {
      const fit = fitRegressionModel(train, 'polynomial', { alpha: 0.1, l1Ratio: 0.5, polynomialDegree: degree, stepwiseTerms: Math.min(degree, 3), knnK: 5, svmC: 1, svmGamma: 1, treeDepth: 4, forestTrees: 35, boostingRounds: 40, learningRate: 0.1, decisionThreshold: 0.5 });
      return computeMetrics(train, fit).mse;
    });
    const validationMse = complexity.map((degree) => {
      const fit = fitRegressionModel(train, 'polynomial', { alpha: 0.1, l1Ratio: 0.5, polynomialDegree: degree, stepwiseTerms: Math.min(degree, 3), knnK: 5, svmC: 1, svmGamma: 1, treeDepth: 4, forestTrees: 35, boostingRounds: 40, learningRate: 0.1, decisionThreshold: 0.5 });
      const yTrue = test.map((point) => point.y);
      const yPred = test.map((point) => fit.predict(getPointFeatures(point)));
      return computeMetricsFromPredictions(yTrue, yPred).mse;
    });
    return { complexity, trainMse, validationMse };
  }

  const alphas = [2.5, 1.5, 1, 0.6, 0.3, 0.1, 0.03];
  const complexity = alphas.map((_, index) => index + 1);
  const trainMse = alphas.map((alpha) => {
    const fit = fitRegressionModel(train, 'ridge', { alpha, l1Ratio: 0.5, polynomialDegree: 2, stepwiseTerms: 2, knnK: 5, svmC: 1, svmGamma: 1, treeDepth: 4, forestTrees: 35, boostingRounds: 40, learningRate: 0.1, decisionThreshold: 0.5 });
    return computeMetrics(train, fit).mse;
  });
  const validationMse = alphas.map((alpha) => {
    const fit = fitRegressionModel(train, 'ridge', { alpha, l1Ratio: 0.5, polynomialDegree: 2, stepwiseTerms: 2, knnK: 5, svmC: 1, svmGamma: 1, treeDepth: 4, forestTrees: 35, boostingRounds: 40, learningRate: 0.1, decisionThreshold: 0.5 });
    const yTrue = test.map((point) => point.y);
    const yPred = test.map((point) => fit.predict(getPointFeatures(point)));
    return computeMetricsFromPredictions(yTrue, yPred).mse;
  });
  return { complexity, trainMse, validationMse };
}

export function computeOLSSolution(data: DataPoint[]): { slope: number; intercept: number } {
  const fit = fitRegressionModel(data, 'ols', {
    alpha: 0,
    l1Ratio: 0.5,
    polynomialDegree: 1,
    stepwiseTerms: 1,
    knnK: 5,
    svmC: 1,
    svmGamma: 1,
    treeDepth: 4,
    forestTrees: 35,
    boostingRounds: 40,
    learningRate: 0.1,
    decisionThreshold: 0.5,
  });
  return {
    slope: fit.coefficients[0] ?? 0,
    intercept: fit.intercept,
  };
}

export function supports2D(modelType: ModelType): boolean {
  return !(
    modelType === 'forward_stepwise'
    || modelType === 'backward_stepwise'
  );
}

export function recommendedDatasets(modelType: ModelType): DatasetType[] {
  const map: Record<ModelType, DatasetType[]> = {
    ols: ['linear', 'noisy'],
    ridge: ['noisy', 'heteroscedastic', 'random_recipe'],
    lasso: ['noisy', 'piecewise', 'random_recipe'],
    elasticnet: ['noisy', 'piecewise', 'random_recipe'],
    polynomial: ['quadratic', 'sinusoidal'],
    forward_stepwise: ['quadratic', 'piecewise'],
    backward_stepwise: ['quadratic', 'piecewise'],
    svm_regressor: ['sinusoidal', 'random_recipe', 'noisy'],
    pcr_regressor: ['noisy', 'heteroscedastic', 'random_recipe'],
    pls_regressor: ['noisy', 'heteroscedastic', 'random_recipe'],
    logistic_classifier: ['class_linear', 'class_imbalanced'],
    knn_classifier: ['class_moons', 'class_overlap'],
    svm_classifier: ['class_overlap', 'class_moons'],
    decision_tree_classifier: ['class_overlap', 'class_linear'],
    random_forest_classifier: ['class_overlap', 'class_imbalanced'],
    adaboost_classifier: ['class_overlap', 'class_imbalanced'],
    gradient_boosting_classifier: ['class_overlap', 'class_imbalanced'],
  };
  return map[modelType] ?? ['random_recipe'];
}

export function latexForModel(modelType: ModelType, params: ModelParams, fit: RegressionFit | null = null): string {
  if (modelType === 'ols') {
    return '\\hat{y} = \\beta_0 + \\beta_1 x';
  }
  if (modelType === 'ridge') {
    return '\\min_{\\beta}\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2 + \\alpha\\sum_{j=1}^{p}\\beta_j^2';
  }
  if (modelType === 'lasso') {
    return '\\min_{\\beta}\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2 + \\alpha\\sum_{j=1}^{p}|\\beta_j|';
  }
  if (modelType === 'elasticnet') {
    return `\\min_{\\beta}\\sum_{i=1}^{n}(y_i-\\hat{y}_i)^2 + ${params.alpha.toFixed(2)}\\left(${params.l1Ratio.toFixed(2)}\\|\\beta\\|_1 + ${(1 - params.l1Ratio).toFixed(2)}\\|\\beta\\|_2^2\\right)`;
  }
  if (modelType === 'polynomial') {
    return `\\hat{y} = \\beta_0 + \\sum_{k=1}^{${params.polynomialDegree}} \\beta_k x^k`;
  }
  if (modelType === 'forward_stepwise' || modelType === 'backward_stepwise') {
    const powers = fit?.featurePowers ?? [];
    const terms = powers.length > 0
      ? powers.map((power) => `\\beta_{${power}}x^{${power}}`).join(' + ')
      : '\\beta_1x';
    return `\\hat{y} = \\beta_0 + ${terms}`;
  }
  if (modelType === 'logistic_classifier') {
    return 'P(y=1|x)=\\sigma(\\beta_0 + \\beta^\\top x)';
  }
  if (modelType === 'knn_classifier') {
    return '\\hat{P}(y=1|x)=\\frac{1}{k}\\sum_{i \\in \\mathcal{N}_k(x)} y_i';
  }
  if (modelType === 'svm_classifier') {
    return '\\hat{y}=\\mathrm{sign}(w^\\top x + b)';
  }
  if (modelType === 'decision_tree_classifier') {
    return '\\hat{P}(y=1|x)=\\text{leaf positive rate}(x)';
  }
  if (modelType === 'random_forest_classifier') {
    return '\\hat{P}(y=1|x)=\\frac{1}{T}\\sum_{t=1}^{T}\\hat{P}_t(y=1|x)';
  }
  if (modelType === 'adaboost_classifier') {
    return '\\hat{y}=\\mathrm{sign}\\left(\\sum_{m=1}^{M}\\alpha_m h_m(x)\\right)';
  }
  if (modelType === 'gradient_boosting_classifier') {
    return '\\hat{P}(y=1|x)=\\sigma\\left(f_0(x)+\\sum_{m=1}^{M}\\eta h_m(x)\\right)';
  }
  if (modelType === 'svm_regressor') {
    return '\\hat{y}=\\sum_i\\alpha_i K(x, x_i)+b';
  }
  if (modelType === 'pcr_regressor') {
    return '\\hat{y}=\\beta_0+\\sum_{k=1}^{m}\\beta_k\\mathrm{PC}_k(x)';
  }
  if (modelType === 'pls_regressor') {
    return '\\hat{y}=\\beta_0+\\sum_{k=1}^{m}\\beta_k t_k(x)';
  }
  return '\\hat{y} = \\beta_0 + \\beta_1 x';
}

export function generatePythonCode(
  modelType: ModelType,
  params: ModelParams,
  evaluationMode: EvaluationMode = 'full',
  testRatio = 0.25,
  cvFolds = 5,
  featureCount = 1
): string {
  if (
    modelType === 'logistic_classifier'
    || modelType === 'knn_classifier'
    || modelType === 'svm_classifier'
    || modelType === 'decision_tree_classifier'
    || modelType === 'random_forest_classifier'
    || modelType === 'adaboost_classifier'
    || modelType === 'gradient_boosting_classifier'
  ) {
    const classMap: Record<string, string> = {
      logistic_classifier: `LogisticRegression(max_iter=1000, C=${(1 / Math.max(params.alpha, 0.01)).toFixed(4)})`,
      knn_classifier: `KNeighborsClassifier(n_neighbors=${Math.max(1, Math.round(params.knnK))})`,
      svm_classifier: `SVC(C=${Math.max(params.svmC, 0.01).toFixed(4)}, gamma=${Math.max(params.svmGamma, 0.01).toFixed(4)}, probability=True)`,
      decision_tree_classifier: `DecisionTreeClassifier(max_depth=${Math.max(1, Math.round(params.treeDepth))}, random_state=42)`,
      random_forest_classifier: `RandomForestClassifier(n_estimators=${Math.max(5, Math.round(params.forestTrees))}, max_depth=${Math.max(1, Math.round(params.treeDepth))}, random_state=42)`,
      adaboost_classifier: `AdaBoostClassifier(n_estimators=${Math.max(5, Math.round(params.boostingRounds))}, learning_rate=${Math.max(params.learningRate, 0.01).toFixed(4)}, random_state=42)`,
      gradient_boosting_classifier: `GradientBoostingClassifier(n_estimators=${Math.max(5, Math.round(params.boostingRounds))}, learning_rate=${Math.max(params.learningRate, 0.01).toFixed(4)}, max_depth=${Math.max(1, Math.round(params.treeDepth))}, random_state=42)`,
    };
    const evaluationBlockCls = evaluationMode === 'train_test'
      ? `from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testRatio.toFixed(2)}, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= ${params.decisionThreshold.toFixed(2)}).astype(int)
y_eval = y_test`
      : evaluationMode === 'cross_validation'
        ? `from sklearn.model_selection import StratifiedKFold, cross_val_predict
cv = StratifiedKFold(n_splits=${cvFolds}, shuffle=True, random_state=42)
y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
y_pred = (y_prob >= ${params.decisionThreshold.toFixed(2)}).astype(int)
y_eval = y`
        : `model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= ${params.decisionThreshold.toFixed(2)}).astype(int)
y_eval = y`;

    return `from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss
import numpy as np

X = np.array(data_x).reshape(-1, ${featureCount})
y = np.array(data_y).astype(int)
model = ${classMap[modelType]}
${evaluationBlockCls}

print(f"Accuracy: {accuracy_score(y_eval, y_pred):.4f}")
print(f"Precision: {precision_score(y_eval, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_eval, y_pred, zero_division=0):.4f}")
print(f"F1: {f1_score(y_eval, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_eval, y_prob):.4f}")
print(f"PR-AUC: {average_precision_score(y_eval, y_prob):.4f}")
print(f"Log Loss: {log_loss(y_eval, y_prob):.4f}")`;
  }

  const evaluationBlock = evaluationMode === 'train_test'
    ? `from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${testRatio.toFixed(2)}, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_eval = y_test`
    : evaluationMode === 'cross_validation'
      ? `from sklearn.model_selection import KFold, cross_val_predict
cv = KFold(n_splits=${cvFolds}, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=cv)
y_eval = y`
      : `model.fit(X, y)
y_pred = model.predict(X)
y_eval = y`;

  if (modelType === 'polynomial') {
    return `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

X = np.array(data_x).reshape(-1, ${featureCount})
y = np.array(data_y)

model = Pipeline([
    ("poly", PolynomialFeatures(degree=${params.polynomialDegree}, include_bias=False)),
    ("regressor", LinearRegression())
])
${evaluationBlock}

print(f"R² Score: {r2_score(y_eval, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_eval, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_eval, y_pred):.4f}")`;
  }

  if (modelType === 'forward_stepwise' || modelType === 'backward_stepwise') {
    const direction = modelType === 'forward_stepwise' ? 'forward' : 'backward';
    return `from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

X_raw = np.array(data_x).reshape(-1, ${featureCount})
y = np.array(data_y)

poly = PolynomialFeatures(degree=${params.polynomialDegree}, include_bias=False)
X = poly.fit_transform(X_raw)
base = LinearRegression()
sfs = SequentialFeatureSelector(
    base,
    n_features_to_select=${params.stepwiseTerms},
    direction="${direction}"
)

X_selected = sfs.fit_transform(X, y)
base.fit(X_selected, y)
y_pred = base.predict(X_selected)

print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")`;
  }

  if (modelType === 'pcr_regressor' || modelType === 'pls_regressor') {
    const comp = modelType === 'pcr_regressor'
      ? Math.max(1, Math.round(params.pcaComponents ?? 2))
      : Math.max(1, Math.round(params.plsComponents ?? 2));
    const transformer = modelType === 'pcr_regressor'
      ? `PCA(n_components=${comp})`
      : `PLSRegression(n_components=${comp})`;
    const reg = modelType === 'pcr_regressor'
      ? `LinearRegression()`
      : `None`;
    return `from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

X = np.array(data_x).reshape(-1, ${featureCount})
y = np.array(data_y)
model = ${modelType === 'pcr_regressor'
  ? `Pipeline([("pca", ${transformer}), ("regressor", ${reg})])`
  : transformer}
${evaluationBlock}

print(f"R² Score: {r2_score(y_eval, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_eval, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_eval, y_pred):.4f}")`;
  }

  const modelMap: Record<string, string> = {
    ols: 'LinearRegression',
    ridge: 'Ridge',
    lasso: 'Lasso',
    elasticnet: 'ElasticNet',
    svm_regressor: 'SVR',
  };
  const className = modelMap[modelType] ?? 'LinearRegression';

  let args = '';
  if (modelType === 'ridge' || modelType === 'lasso') {
    args = `alpha=${params.alpha.toFixed(4)}`;
  }
  if (modelType === 'elasticnet') {
    args = `alpha=${params.alpha.toFixed(4)}, l1_ratio=${params.l1Ratio.toFixed(4)}`;
  }

  if (modelType === 'svm_regressor') {
    args = `C=${Math.max(params.svmC, 0.01).toFixed(4)}, gamma=${Math.max(params.svmGamma, 0.01).toFixed(4)}, epsilon=${Math.max(params.svmEpsilon ?? 0.1, 0.01).toFixed(4)}`;
  }

  return `from sklearn.linear_model import ${className}
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

X = np.array(data_x).reshape(-1, ${featureCount})
y = np.array(data_y)
model = ${className}(${args})
${evaluationBlock}

print(f"R² Score: {r2_score(y_eval, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_eval, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_eval, y_pred):.4f}")`;
}
