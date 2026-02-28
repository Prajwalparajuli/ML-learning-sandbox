import type { CnnState, GodModeState, HudPulse, MlpState } from './godModeTypes';

interface NarrationFrame {
  title: string;
  body: string;
  pulse: HudPulse;
}

function paint(token: string, color: string): string {
  return `<span style="color:${color};font-weight:700">${token}</span>`;
}

function toFixed(value: number): string {
  return value.toFixed(3);
}

export function buildMlpNarration(mlp: MlpState, colors: { w1: string; w2: string }): NarrationFrame {
  const w1 = paint('w1', colors.w1);
  const w2 = paint('w2', colors.w2);

  if (mlp.step <= 0) {
    return {
      title: 'MLP Input State',
      body: `Deterministic input x=[${mlp.x[0].toFixed(1)}, ${mlp.x[1].toFixed(1)}], bias=${mlp.b.toFixed(1)}, target=${mlp.target.toFixed(1)}.`,
      pulse: 'none',
    };
  }
  if (mlp.step === 1) {
    return {
      title: 'Weighted Summation',
      body: `z = ${w1}*x1 + ${w2}*x2 + b = ${toFixed(mlp.z)}. Drag edge up/down to change ${w1} and ${w2}.`,
      pulse: 'none',
    };
  }
  if (mlp.step === 2) {
    return {
      title: 'ReLU Fold Manifold',
      body: `a = ReLU(z) = ${toFixed(mlp.a)}. Negative half of the manifold folds to zero.`,
      pulse: 'none',
    };
  }
  if (mlp.step === 3) {
    return {
      title: 'Output And Loss',
      body: `y = ${toFixed(mlp.y)}, loss = 0.5*(y-target)^2 = ${toFixed(mlp.loss)}.`,
      pulse: 'none',
    };
  }
  if (mlp.step === 4) {
    return {
      title: 'Backprop Gradient',
      body: `dL/d${w1} and dL/d${w2} are active. Press Backpropagate or Step once more for one gradient descent move.`,
      pulse: 'none',
    };
  }
  return {
    title: 'Loss Landscape Update',
    body: `Ball rolls down the wireframe bowl using current learning rate.`,
    pulse: 'none',
  };
}

export function buildCnnNarration(cnn: CnnState): NarrationFrame {
  if (cnn.stage === 'input') {
    return {
      title: 'Input Image',
      body: '6x6 deterministic matrix with vertical structures ready for scanning.',
      pulse: 'none',
    };
  }

  if (cnn.stage === 'conv') {
    const match = cnn.matchScore >= 1.5;
    return {
      title: 'Conv1 Flashlight Frustum',
      body: `Kernel window at row ${cnn.scan.row}, col ${cnn.scan.col}. Detector icon: <span class="studio-god-detector-icon">|</span>. Sum(x*w) = ${cnn.matchScore.toFixed(3)}${match ? ' -> vertical edge match' : ''}.`,
      pulse: match ? 'green' : 'none',
    };
  }

  if (cnn.stage === 'pool') {
    return {
      title: 'Pooling Plane',
      body: '2x2 max pooling keeps strongest local responses.',
      pulse: 'none',
    };
  }

  if (cnn.stage === 'flatten') {
    return {
      title: 'Flatten Unzip',
      body: 'Pooled 2D map collapses into one 1D vertical feature array.',
      pulse: 'none',
    };
  }

  return {
    title: 'Dense Decision',
    body: `Softmax output: p(edge)=${(cnn.dense[0] * 100).toFixed(1)}%, p(other)=${(cnn.dense[1] * 100).toFixed(1)}%.`,
    pulse: 'none',
  };
}

export function buildNarration(state: GodModeState, colors: { w1: string; w2: string }): NarrationFrame {
  return state.mode === 'mlp' ? buildMlpNarration(state.mlp, colors) : buildCnnNarration(state.cnn);
}
