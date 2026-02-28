import {
  clamp,
  cnnScanFromStep,
  cnnStageFromStep,
  CNN_INPUT,
  CNN_KERNEL,
  computeMlpForward,
  computeMlpGradient,
  evaluateCnn,
  lerp,
  MLP_BIAS,
  MLP_INITIAL_W,
  MLP_MAX_STEP,
  MLP_TARGET,
  MLP_X,
  weightColor,
  CNN_MAX_STEP,
} from './dummyData';
import { buildNarration } from './narrationEngine';
import type { CnnState, GodModeAction, GodModeState, GodModeStore, MlpState, Playback, StoreListener } from './godModeTypes';

const DEFAULT_LR = 0.08;

function createMlpState(): MlpState {
  const w: [number, number] = [MLP_INITIAL_W[0], MLP_INITIAL_W[1]];
  const forward = computeMlpForward(w, MLP_BIAS, MLP_X, MLP_TARGET);
  return {
    x: [MLP_X[0], MLP_X[1]],
    w,
    b: MLP_BIAS,
    z: forward.z,
    a: forward.a,
    y: forward.y,
    target: MLP_TARGET,
    loss: forward.loss,
    dragEdge: null,
    foldT: 0,
    landscape: {
      ball: { w1: w[0], w2: w[1] },
      targetBall: { w1: w[0], w2: w[1] },
      animT: 1,
      active: false,
    },
    step: 0,
  };
}

function createCnnState(): CnnState {
  const scan = cnnScanFromStep(0);
  const evalResult = evaluateCnn(CNN_INPUT, CNN_KERNEL, scan.row, scan.col);
  return {
    input: CNN_INPUT.map((row) => row.slice()),
    kernel: CNN_KERNEL.map((row) => row.slice()),
    conv: evalResult.conv,
    pool: evalResult.pool,
    flat: evalResult.flat,
    dense: evalResult.dense,
    scan,
    stage: cnnStageFromStep(0),
    matchScore: evalResult.matchScore,
    frustumT: 0,
    unzipT: 0,
    step: 0,
  };
}

function hudAnchorForMode(state: GodModeState): { x: number; y: number } {
  if (state.mode === 'mlp') {
    return { x: 58, y: 22 };
  }
  if (state.cnn.stage === 'conv') {
    return {
      x: 16 + state.cnn.scan.col * 12,
      y: 26 + state.cnn.scan.row * 8,
    };
  }
  if (state.cnn.stage === 'pool') return { x: 45, y: 30 };
  if (state.cnn.stage === 'flatten') return { x: 62, y: 28 };
  return { x: 80, y: 24 };
}

function withDerived(next: GodModeState): GodModeState {
  const mlpForward = computeMlpForward(next.mlp.w, next.mlp.b, next.mlp.x, next.mlp.target);
  const mlp = {
    ...next.mlp,
    z: mlpForward.z,
    a: mlpForward.a,
    y: mlpForward.y,
    loss: mlpForward.loss,
  };

  const scan = cnnScanFromStep(next.cnn.step);
  const stage = cnnStageFromStep(next.cnn.step);
  const evalResult = evaluateCnn(next.cnn.input, next.cnn.kernel, scan.row, scan.col);
  const cnn = {
    ...next.cnn,
    scan,
    stage,
    conv: evalResult.conv,
    pool: evalResult.pool,
    flat: evalResult.flat,
    dense: evalResult.dense,
    matchScore: evalResult.matchScore,
  };

  return {
    ...next,
    mlp,
    cnn,
    globalStep: next.mode === 'mlp' ? mlp.step : cnn.step,
  };
}

function withNarration(next: GodModeState): GodModeState {
  const colors = {
    w1: weightColor(next.mlp.w[0]),
    w2: weightColor(next.mlp.w[1]),
  };
  const frame = buildNarration(next, colors);
  const anchor = hudAnchorForMode(next);
  return {
    ...next,
    hud: {
      ...next.hud,
      visible: true,
      title: frame.title,
      body: frame.body,
      pulse: frame.pulse,
      x: anchor.x,
      y: anchor.y,
      colorBindings: {
        w1: colors.w1,
        w2: colors.w2,
      },
    },
  };
}

function applyBackprop(state: GodModeState): GodModeState {
  const gradient = computeMlpGradient(state.mlp.w, state.mlp.b, state.mlp.x, state.mlp.target);
  const nextW1 = clamp(state.mlp.w[0] - state.learningRate * gradient.dw1, -1.5, 1.5);
  const nextW2 = clamp(state.mlp.w[1] - state.learningRate * gradient.dw2, -1.5, 1.5);
  return {
    ...state,
    mlp: {
      ...state.mlp,
      w: [nextW1, nextW2],
      step: MLP_MAX_STEP,
      landscape: {
        ball: {
          w1: state.mlp.landscape.active
            ? lerp(state.mlp.landscape.ball.w1, state.mlp.landscape.targetBall.w1, state.mlp.landscape.animT)
            : state.mlp.w[0],
          w2: state.mlp.landscape.active
            ? lerp(state.mlp.landscape.ball.w2, state.mlp.landscape.targetBall.w2, state.mlp.landscape.animT)
            : state.mlp.w[1],
        },
        targetBall: { w1: nextW1, w2: nextW2 },
        animT: 0,
        active: true,
      },
    },
  };
}

function resetForMode(state: GodModeState): GodModeState {
  if (state.mode === 'mlp') {
    return {
      ...state,
      playback: 'idle',
      mlp: createMlpState(),
    };
  }

  return {
    ...state,
    playback: 'idle',
    cnn: createCnnState(),
  };
}

function tickState(state: GodModeState, dtMs: number): GodModeState {
  const foldTarget = state.mlp.z < 0 ? 1 : 0;
  let foldT = state.mlp.foldT;
  const foldSpeed = clamp(dtMs / 350, 0, 1);
  foldT = foldT + (foldTarget - foldT) * foldSpeed;

  let landscape = state.mlp.landscape;
  if (landscape.active) {
    const nextT = clamp(landscape.animT + dtMs / 450, 0, 1);
    landscape = {
      ...landscape,
      animT: nextT,
      active: nextT < 1,
      ball: nextT >= 1 ? { ...landscape.targetBall } : landscape.ball,
    };
  }

  const frustumTarget = state.cnn.stage === 'conv' ? 1 : 0;
  const unzipTarget = state.cnn.stage === 'flatten' || state.cnn.stage === 'dense' ? 1 : 0;

  const frustumT = state.cnn.frustumT + (frustumTarget - state.cnn.frustumT) * clamp(dtMs / 220, 0, 1);
  const unzipT = state.cnn.unzipT + (unzipTarget - state.cnn.unzipT) * clamp(dtMs / 260, 0, 1);

  return {
    ...state,
    mlp: {
      ...state.mlp,
      foldT,
      landscape,
    },
    cnn: {
      ...state.cnn,
      frustumT,
      unzipT,
    },
  };
}

function reduceState(state: GodModeState, action: GodModeAction): GodModeState {
  switch (action.type) {
    case 'mode/set': {
      return {
        ...state,
        mode: action.mode,
        playback: 'idle',
      };
    }
    case 'playback/toggle': {
      const nextPlayback: Playback = state.playback === 'playing' ? 'paused' : 'playing';
      return {
        ...state,
        playback: nextPlayback,
      };
    }
    case 'playback/set': {
      return {
        ...state,
        playback: action.playback,
      };
    }
    case 'reset': {
      return resetForMode(state);
    }
    case 'step': {
      if (state.mode === 'mlp') {
        const nextStep = clamp(state.mlp.step + 1, 0, MLP_MAX_STEP);
        const stepped = {
          ...state,
          mlp: {
            ...state.mlp,
            step: nextStep,
          },
        };
        if (nextStep === MLP_MAX_STEP) return applyBackprop(stepped);
        return stepped;
      }

      const nextStep = clamp(state.cnn.step + 1, 0, CNN_MAX_STEP);
      return {
        ...state,
        cnn: {
          ...state.cnn,
          step: nextStep,
        },
      };
    }
    case 'learningRate/set': {
      return {
        ...state,
        learningRate: clamp(action.value, 0.01, 0.2),
      };
    }
    case 'tick': {
      return tickState(state, action.dtMs);
    }
    case 'mlp/setDragEdge': {
      return {
        ...state,
        mlp: {
          ...state.mlp,
          dragEdge: action.edge,
          step: Math.max(state.mlp.step, 1),
        },
      };
    }
    case 'mlp/dragWeight': {
      const delta = -action.deltaY * 0.008;
      const nextWeight = clamp(state.mlp.w[action.edge] + delta, -1.5, 1.5);
      const nextW: [number, number] = [...state.mlp.w] as [number, number];
      nextW[action.edge] = nextWeight;
      return {
        ...state,
        mlp: {
          ...state.mlp,
          w: nextW,
          dragEdge: action.edge,
          step: Math.max(state.mlp.step, 1),
        },
      };
    }
    case 'mlp/backprop': {
      return applyBackprop(state);
    }
    default:
      return state;
  }
}

function createInitialState(): GodModeState {
  const state: GodModeState = {
    mode: 'mlp',
    playback: 'idle',
    globalStep: 0,
    learningRate: DEFAULT_LR,
    hud: {
      visible: true,
      x: 58,
      y: 22,
      title: 'God Mode Learn Tab',
      body: 'Interactive state machine ready.',
      colorBindings: {},
      pulse: 'none',
    },
    mlp: createMlpState(),
    cnn: createCnnState(),
  };

  return withNarration(withDerived(state));
}

export function createGodModeStore(): GodModeStore {
  let state = createInitialState();
  const listeners = new Set<StoreListener>();

  const notify = () => {
    listeners.forEach((listener) => listener(state));
  };

  return {
    getState: () => state,
    dispatch: (action: GodModeAction) => {
      const reduced = reduceState(state, action);
      const withComputed = withDerived(reduced);
      state = action.type === 'tick' ? withComputed : withNarration(withComputed);
      notify();
    },
    subscribe: (listener: StoreListener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}
