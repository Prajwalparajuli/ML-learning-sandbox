export interface DeepDefaults {
  sample_image_id: number;
  default_filter: 'horizontal_edge' | 'vertical_edge';
}

export interface DeepImageRecord {
  id: number;
  key: string;
  label: string;
  class: 'cat' | 'dog';
  file: string;
}

export interface DeepLearningManifest {
  defaults: DeepDefaults;
  images: DeepImageRecord[];
}

export interface DeepResolvedImageRecord extends DeepImageRecord {
  src: string;
}

export interface DeepLearningResolvedManifest {
  defaults: DeepDefaults;
  images: DeepResolvedImageRecord[];
}

export interface MlpStateRecord {
  id: string;
  hiddenLayers: number;
  neuronsPerLayer: number;
  activation: 'relu' | 'sigmoid';
  accuracy?: number;
}

export interface CnnSweepRecord {
  id: string;
  filterType: 'horizontal_edge' | 'vertical_edge';
  stepCount: number;
}

export interface TrainingReplayRecord {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface TrainingReplayPoint {
  epoch: number;
  training_loss: number;
  validation_loss: number;
  training_accuracy: number;
  validation_accuracy: number;
  cat_confidence: number;
  dog_confidence: number;
}

export interface ModelComparisonRecord {
  id: string;
  label: string;
  params_millions: number;
  top1_accuracy: number;
  latency_ms: number;
  note: string;
}

export interface ParameterEffectRecord {
  id: string;
  parameter: string;
  setting: string;
  expected_effect: string;
  visual_signal: string;
  category: 'instant_visual' | 'precomputed_effect';
}

export type DeepModule = 'mlp' | 'cnn';
export type DeepViewMode = 'heatmap' | 'numbers';
export type DeepDensityMode = 'focused_single_stage';
export type DeepExperienceMode = 'real_inference' | 'kernel_lab' | 'legacy';
export type DeepRuntimeBackend = 'webgl' | 'wasm' | 'cpu';
export type DeepModelKind = 'mnist_mlp' | 'mnist_cnn' | 'boundary_shallow' | 'boundary_deep';
export type DeepVizDensity = 'balanced' | 'high' | 'minimal';
export type DlInferenceMode = 'pure' | 'assisted';

export interface DeepStepDelta {
  metric: 'z' | 'activation' | 'confidence';
  previous: number;
  current: number;
  delta: number;
}

export interface DeepNarrationItem {
  step_id: string;
  what_changed: string;
  why: string;
  try_next: string;
  misconception: string;
  predict_prompt?: string;
  reveal_text?: string;
}

export interface DeepCheckpoint {
  id: string;
  label: string;
  completed: boolean;
}

export interface InferenceSnapshot {
  logits: number[];
  probabilities: number[];
  predictedClass: number;
  latencyMs: number;
}

export interface MlpContributionEdge {
  sourceLayer: 'input_group' | 'hidden';
  sourceIndex: number;
  targetLayer: 'hidden' | 'output';
  targetIndex: number;
  contribution: number;
}

export interface MlpActivationTrace {
  inputGroups: number[];
  hiddenActivations: number[];
  outputLogits: number[];
  contributionEdges: MlpContributionEdge[];
}

export interface CnnMapSlice {
  channel: number;
  values: number[][];
  meanActivation: number;
}

export interface CnnActivationTrace {
  conv: CnnMapSlice[];
  relu: CnnMapSlice[];
  pool: CnnMapSlice[];
  probabilities: number[];
}

export interface DlEvalMetrics {
  accuracy: number;
  precisionMacro: number;
  recallMacro: number;
  confusion10x10: number[][];
  sampleCount: number;
}

export interface MlpStepMath {
  inputVector: number[];
  weightVector: number[];
  bias: number;
  z: number;
  activationOutput: number;
  activationPoint: { z: number; a: number };
  logits: [number, number];
  probabilities: { cat: number; dog: number };
  outputConfidenceGap: number;
}

export interface CnnPipelineStage {
  id: 'input' | 'conv' | 'relu' | 'pool' | 'flatten' | 'dense';
  label: string;
  story_label: string;
  summary: string;
  stage_math_summary?: string;
  stage_visual_payload?: { type: 'matrix' | 'vector' | 'bars' | 'image'; note: string };
  predict_prompt?: string;
  reveal_text?: string;
  misconception: string;
}
