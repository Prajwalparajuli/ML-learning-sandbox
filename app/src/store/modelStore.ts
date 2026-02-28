import { create } from 'zustand';

export type ModelType =
  | 'ols'
  | 'ridge'
  | 'lasso'
  | 'elasticnet'
  | 'polynomial'
  | 'forward_stepwise'
  | 'backward_stepwise'
  | 'svm_regressor'
  | 'pcr_regressor'
  | 'pls_regressor'
  | 'logistic_classifier'
  | 'knn_classifier'
  | 'svm_classifier'
  | 'decision_tree_classifier'
  | 'random_forest_classifier'
  | 'adaboost_classifier'
  | 'gradient_boosting_classifier';
export type DatasetType =
  | 'linear'
  | 'noisy'
  | 'outliers'
  | 'heteroscedastic'
  | 'quadratic'
  | 'sinusoidal'
  | 'piecewise'
  | 'class_linear'
  | 'class_overlap'
  | 'class_moons'
  | 'class_imbalanced'
  | 'random_recipe';
export type TaskMode = 'regression' | 'classification' | 'deep_learning';
export type OnboardingState = 'not_started' | 'in_progress' | 'completed';
export type MissionTier = 'starter' | 'analyst' | 'expert';

export interface RandomDataRecipe {
  pattern: 'linear' | 'polynomial' | 'sinusoidal' | 'piecewise' | 'mixed';
  noiseType: 'gaussian' | 'heteroscedastic' | 'heavy_tail';
  outlierLevel: 'none' | 'low' | 'medium' | 'high';
  correlatedFeatures: boolean;
}

export interface ModelParams {
  alpha: number;  // Regularization strength
  l1Ratio: number; // For ElasticNet (0=Ridge, 1=Lasso)
  polynomialDegree: number;
  stepwiseTerms: number;
  knnK: number;
  svmC: number;
  svmGamma: number;
  svmEpsilon?: number;
  treeDepth: number;
  forestTrees: number;
  boostingRounds: number;
  learningRate: number;
  pcaComponents?: number;
  plsComponents?: number;
  decisionThreshold: number;
}

export interface Metrics {
  r2: number;
  rmse: number;
  mae: number;
  mse: number;
  mape: number;
  explainedVariance: number;
  medianAe: number;
  adjustedR2: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  rocAuc: number;
  prAuc: number;
  logLoss: number;
}

export type MetricKey = keyof Metrics;

export interface DataPoint {
  x: number;
  x2?: number;
  features: number[];
  y: number;
}

export type EvaluationMode = 'full' | 'train_test' | 'cross_validation';
export type FeatureMode = '1d' | '2d';
export type HeroLayoutMode = 'compact' | 'expanded';
export type ViewMode = 'focus' | 'deep_dive';

export interface SandboxSnapshot {
  taskMode: TaskMode;
  modelType: ModelType;
  dataset: DatasetType;
  params: ModelParams;
  evaluationMode: EvaluationMode;
  testRatio: number;
  cvFolds: number;
  featureMode: FeatureMode;
  selectedMetrics: MetricKey[];
  sampleSize: number;
  randomSeed: number;
  datasetVersion: number;
}

interface ModelState {
  taskMode: TaskMode;
  setTaskMode: (mode: TaskMode) => void;
  // Model selection
  modelType: ModelType;
  setModelType: (type: ModelType) => void;
  
  // Dataset selection
  dataset: DatasetType;
  setDataset: (dataset: DatasetType) => void;
  
  // Parameters
  params: ModelParams;
  setParam: <K extends keyof ModelParams>(key: K, value: ModelParams[K]) => void;
  setParams: (params: Partial<ModelParams>) => void;
  
  // Metrics
  metrics: Metrics;
  setMetrics: (metrics: Metrics) => void;
  
  // Data
  data: DataPoint[];
  setData: (data: DataPoint[]) => void;

  // Evaluation
  evaluationMode: EvaluationMode;
  setEvaluationMode: (mode: EvaluationMode) => void;
  testRatio: number;
  setTestRatio: (ratio: number) => void;
  cvFolds: number;
  setCvFolds: (folds: number) => void;
  featureMode: FeatureMode;
  setFeatureMode: (mode: FeatureMode) => void;
  selectedMetrics: MetricKey[];
  setSelectedMetrics: (metrics: MetricKey[]) => void;
  heroLayoutMode: HeroLayoutMode;
  setHeroLayoutMode: (mode: HeroLayoutMode) => void;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  onboardingState: OnboardingState;
  setOnboardingState: (state: OnboardingState) => void;
  missionTier: MissionTier;
  setMissionTier: (tier: MissionTier) => void;
  randomDataRecipe: RandomDataRecipe;
  setRandomDataRecipe: (recipe: Partial<RandomDataRecipe>) => void;
  generateRandomDatasetFromRecipe: () => void;
  
  // UI State
  showAssumptions: boolean;
  setShowAssumptions: (show: boolean) => void;
  showClassificationDiagnostics: boolean;
  setShowClassificationDiagnostics: (show: boolean) => void;
  showOlsSolution: boolean;
  setShowOlsSolution: (show: boolean) => void;
  compareWithOls: boolean;
  setCompareWithOls: (show: boolean) => void;
  comparePinnedModels: ModelType[];
  togglePinnedModel: (model: ModelType) => void;
  clearPinnedModels: () => void;
  sampleSize: number;
  setSampleSize: (size: number) => void;
  randomSeed: number;
  setRandomSeed: (seed: number) => void;
  datasetVersion: number;
  regenerateDataset: () => void;
  resetParams: () => void;
  captureSandboxSnapshot: () => SandboxSnapshot;
  restoreSandboxSnapshot: (snapshot: SandboxSnapshot) => void;
  
  // Error handling
  error: string | null;
  setError: (error: string | null) => void;
}

export const useModelStore = create<ModelState>((set, get) => ({
  taskMode: 'regression',
  setTaskMode: (mode) => set((state) => {
    if (mode === 'deep_learning') {
      return { taskMode: mode };
    }

    return {
      taskMode: mode,
      modelType: mode === 'regression' ? 'ols' : 'logistic_classifier',
      dataset: mode === 'regression' ? 'linear' : 'class_moons',
      evaluationMode: mode === 'classification' ? 'train_test' : state.evaluationMode,
      featureMode: mode === 'classification' ? '2d' : state.featureMode,
      heroLayoutMode: 'compact',
      viewMode: 'deep_dive',
      selectedMetrics: mode === 'regression'
        ? ['r2', 'rmse', 'mae', 'mse']
        : ['accuracy', 'f1', 'precision', 'recall'],
      showAssumptions: mode === 'classification' ? false : state.showAssumptions,
      showClassificationDiagnostics: mode === 'classification' ? true : state.showClassificationDiagnostics,
      showOlsSolution: mode === 'classification' ? false : state.showOlsSolution,
      compareWithOls: mode === 'classification' ? false : state.compareWithOls,
      params: {
        ...state.params,
        decisionThreshold: 0.5,
        knnK: 5,
        svmC: 1,
        svmGamma: 1,
        svmEpsilon: 0.1,
        treeDepth: 4,
        forestTrees: 35,
        boostingRounds: 40,
        learningRate: 0.1,
        pcaComponents: 2,
        plsComponents: 2,
      },
    };
  }),

  modelType: 'ols',
  setModelType: (type) => set({ modelType: type }),
  
  dataset: 'linear',
  setDataset: (dataset) => set({ dataset }),
  
  params: {
    alpha: 0.1,
    l1Ratio: 0.5,
    polynomialDegree: 3,
    stepwiseTerms: 2,
    knnK: 5,
    svmC: 1,
    svmGamma: 1,
    svmEpsilon: 0.1,
    treeDepth: 4,
    forestTrees: 35,
    boostingRounds: 40,
    learningRate: 0.1,
    pcaComponents: 2,
    plsComponents: 2,
    decisionThreshold: 0.5,
  },
  setParam: (key, value) => set((state) => ({
    params: { ...state.params, [key]: value }
  })),
  setParams: (newParams) => set((state) => ({
    params: { ...state.params, ...newParams }
  })),
  
  metrics: {
    r2: 0,
    rmse: 0,
    mae: 0,
    mse: 0,
    mape: 0,
    explainedVariance: 0,
    medianAe: 0,
    adjustedR2: 0,
    accuracy: 0,
    precision: 0,
    recall: 0,
    specificity: 0,
    f1: 0,
    rocAuc: 0,
    prAuc: 0,
    logLoss: 0,
  },
  setMetrics: (metrics) => set({ metrics }),
  
  data: [],
  setData: (data) => set({ data }),

  evaluationMode: 'train_test',
  setEvaluationMode: (mode) => set({ evaluationMode: mode }),
  testRatio: 0.25,
  setTestRatio: (ratio) => set({ testRatio: ratio }),
  cvFolds: 5,
  setCvFolds: (folds) => set({ cvFolds: folds }),
  featureMode: '1d',
  setFeatureMode: (mode) => set({ featureMode: mode }),
  selectedMetrics: ['r2', 'rmse', 'mae', 'mse'],
  setSelectedMetrics: (metrics) => set({ selectedMetrics: metrics }),
  heroLayoutMode: 'compact',
  setHeroLayoutMode: (mode) => set({ heroLayoutMode: mode }),
  viewMode: 'deep_dive',
  setViewMode: (mode) => set({ viewMode: mode }),
  onboardingState: 'not_started',
  setOnboardingState: (onboardingState) => set({ onboardingState }),
  missionTier: 'starter',
  setMissionTier: (missionTier) => set({ missionTier }),
  randomDataRecipe: {
    pattern: 'linear',
    noiseType: 'gaussian',
    outlierLevel: 'none',
    correlatedFeatures: true,
  },
  setRandomDataRecipe: (recipe) => set((state) => ({ randomDataRecipe: { ...state.randomDataRecipe, ...recipe } })),
  generateRandomDatasetFromRecipe: () => set((state) => ({
    dataset: 'random_recipe',
    datasetVersion: state.datasetVersion + 1,
  })),
  
  showAssumptions: false,
  setShowAssumptions: (show) => set({ showAssumptions: show }),
  showClassificationDiagnostics: true,
  setShowClassificationDiagnostics: (show) => set({ showClassificationDiagnostics: show }),
  showOlsSolution: false,
  setShowOlsSolution: (show) => set({ showOlsSolution: show }),
  compareWithOls: false,
  setCompareWithOls: (show) => set({ compareWithOls: show }),
  comparePinnedModels: [],
  togglePinnedModel: (model) =>
    set((state) => {
      if (state.comparePinnedModels.includes(model)) {
        return { comparePinnedModels: state.comparePinnedModels.filter((entry) => entry !== model) };
      }
      if (state.comparePinnedModels.length >= 2) {
        return { comparePinnedModels: [...state.comparePinnedModels.slice(1), model] };
      }
      return { comparePinnedModels: [...state.comparePinnedModels, model] };
    }),
  clearPinnedModels: () => set({ comparePinnedModels: [] }),
  sampleSize: 50,
  setSampleSize: (size) => set({ sampleSize: size }),
  randomSeed: 42,
  setRandomSeed: (seed) => set({ randomSeed: seed }),
  datasetVersion: 0,
  regenerateDataset: () => set((state) => ({ datasetVersion: state.datasetVersion + 1 })),
  resetParams: () => set((state) => ({
    params: {
      ...state.params,
      alpha: 0.1,
      l1Ratio: 0.5,
      polynomialDegree: 3,
      stepwiseTerms: 2,
      knnK: 5,
      svmC: 1,
      svmGamma: 1,
      svmEpsilon: 0.1,
      treeDepth: 4,
      forestTrees: 35,
      boostingRounds: 40,
      learningRate: 0.1,
      pcaComponents: 2,
      plsComponents: 2,
      decisionThreshold: 0.5,
    },
  })),
  captureSandboxSnapshot: (): SandboxSnapshot => {
    const state = get();
    return {
      taskMode: state.taskMode,
      modelType: state.modelType,
      dataset: state.dataset,
      params: { ...state.params },
      evaluationMode: state.evaluationMode,
      testRatio: state.testRatio,
      cvFolds: state.cvFolds,
      featureMode: state.featureMode,
      selectedMetrics: [...state.selectedMetrics],
      sampleSize: state.sampleSize,
      randomSeed: state.randomSeed,
      datasetVersion: state.datasetVersion,
    };
  },
  restoreSandboxSnapshot: (snapshot: SandboxSnapshot) =>
    set({
      taskMode: snapshot.taskMode,
      modelType: snapshot.modelType,
      dataset: snapshot.dataset,
      params: { ...snapshot.params },
      evaluationMode: snapshot.evaluationMode,
      testRatio: snapshot.testRatio,
      cvFolds: snapshot.cvFolds,
      featureMode: snapshot.featureMode,
      selectedMetrics: [...snapshot.selectedMetrics],
      sampleSize: snapshot.sampleSize,
      randomSeed: snapshot.randomSeed,
      datasetVersion: snapshot.datasetVersion,
    }),
  
  error: null,
  setError: (error) => set({ error }),
}));
