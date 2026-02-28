export type LearnMode = 'mlp' | 'cnn';

export type Playback = 'idle' | 'playing' | 'paused';

export type HudPulse = 'none' | 'green';

export type CnnStage = 'input' | 'conv' | 'pool' | 'flatten' | 'dense';

export interface HudState {
  visible: boolean;
  x: number;
  y: number;
  title: string;
  body: string;
  colorBindings: Record<string, string>;
  pulse: HudPulse;
}

export interface MlpLandscape {
  ball: { w1: number; w2: number };
  targetBall: { w1: number; w2: number };
  animT: number;
  active: boolean;
}

export interface MlpState {
  x: [number, number];
  w: [number, number];
  b: number;
  z: number;
  a: number;
  y: number;
  target: number;
  loss: number;
  dragEdge: 0 | 1 | null;
  foldT: number;
  landscape: MlpLandscape;
  step: number;
}

export interface CnnState {
  input: number[][];
  kernel: number[][];
  conv: number[][];
  pool: number[][];
  flat: number[];
  dense: number[];
  scan: { row: number; col: number };
  stage: CnnStage;
  matchScore: number;
  frustumT: number;
  unzipT: number;
  step: number;
}

export interface GodModeState {
  mode: LearnMode;
  playback: Playback;
  globalStep: number;
  learningRate: number;
  hud: HudState;
  mlp: MlpState;
  cnn: CnnState;
}

export type GodModeAction =
  | { type: 'mode/set'; mode: LearnMode }
  | { type: 'playback/toggle' }
  | { type: 'playback/set'; playback: Playback }
  | { type: 'reset' }
  | { type: 'step' }
  | { type: 'learningRate/set'; value: number }
  | { type: 'tick'; dtMs: number }
  | { type: 'mlp/setDragEdge'; edge: 0 | 1 | null }
  | { type: 'mlp/dragWeight'; edge: 0 | 1; deltaY: number }
  | { type: 'mlp/backprop' };

export type StoreListener = (state: GodModeState) => void;

export interface GodModeStore {
  getState: () => GodModeState;
  dispatch: (action: GodModeAction) => void;
  subscribe: (listener: StoreListener) => () => void;
}
