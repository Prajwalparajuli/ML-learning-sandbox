import { useMemo, useState } from 'react';
import { InlineMath } from 'react-katex';
import { flattenImageMatrix } from '@/lib/deepLearning/mnistPreprocess';

type LearnTaskMode = 'classification' | 'regression';
type LearnActivation = 'relu' | 'tanh';

interface LearnWeights {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
  w3c: number[][];
  b3c: number[];
  w3r: number[][];
  b3r: number[];
}

interface LearnGradients {
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
  w3: number[][];
  b3: number[];
}

interface LearnPassResult {
  z1: number[];
  a1: number[];
  z2: number[];
  a2: number[];
  z3: number[];
  output: number[];
  loss: number;
  predictedClass: number;
  gradients: LearnGradients;
}

interface NeuralNetworkLearnLabProps {
  image: number[][];
}

const INPUT_DIM = 5;
const HIDDEN_1 = 5;
const HIDDEN_2 = 4;
const CLASS_COUNT = 10;

const featureLabels = [
  'Ink density',
  'X center offset',
  'Y center offset',
  'BBox area ratio',
  'Left-right symmetry',
];

const stageLabels = [
  'Input',
  'Hidden 1',
  'Hidden 2',
  'Output',
  'Loss',
  'Backprop',
];

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const pseudo = (index: number): number => {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
};

const makeMatrix = (rows: number, cols: number, scale: number, seedBase: number): number[][] =>
  Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => (pseudo(seedBase + r * cols + c) - 0.5) * scale)
  );

const makeVector = (size: number, scale: number, seedBase: number): number[] =>
  Array.from({ length: size }, (_, i) => (pseudo(seedBase + i) - 0.5) * scale);

const createInitialWeights = (): LearnWeights => ({
  w1: makeMatrix(INPUT_DIM, HIDDEN_1, 0.32, 11),
  b1: makeVector(HIDDEN_1, 0.08, 101),
  w2: makeMatrix(HIDDEN_1, HIDDEN_2, 0.28, 211),
  b2: makeVector(HIDDEN_2, 0.08, 311),
  w3c: makeMatrix(HIDDEN_2, CLASS_COUNT, 0.24, 401),
  b3c: makeVector(CLASS_COUNT, 0.05, 601),
  w3r: makeMatrix(HIDDEN_2, 1, 0.26, 701),
  b3r: makeVector(1, 0.05, 801),
});

const activation = (value: number, kind: LearnActivation): number =>
  kind === 'relu' ? (value > 0 ? value : 0) : Math.tanh(value);

const activationPrime = (z: number, a: number, kind: LearnActivation): number => {
  if (kind === 'relu') return z > 0 ? 1 : 0;
  return 1 - a * a;
};

const softmax = (values: number[]): number[] => {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / Math.max(sum, 1e-10));
};

const argMax = (values: number[]): number => {
  let bestIndex = 0;
  let bestValue = values[0] ?? Number.NEGATIVE_INFINITY;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
};

const l2NormMatrix = (matrix: number[][]): number => {
  let sum = 0;
  for (let r = 0; r < matrix.length; r += 1) {
    for (let c = 0; c < matrix[r].length; c += 1) sum += matrix[r][c] * matrix[r][c];
  }
  return Math.sqrt(sum);
};

const l2NormVector = (vector: number[]): number => {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) sum += vector[i] * vector[i];
  return Math.sqrt(sum);
};

const extractLearnFeatures = (image: number[][]): number[] => {
  const flat = flattenImageMatrix(image);
  const signalMask = flat.map((value) => (value > 0.08 ? 1 : 0));
  const pixelCount = flat.length;
  const totalInk = flat.reduce((sum, value) => sum + value, 0);
  const density = totalInk / Math.max(1, pixelCount);

  let weightedRow = 0;
  let weightedCol = 0;
  let mass = 0;
  let minRow = 27;
  let minCol = 27;
  let maxRow = 0;
  let maxCol = 0;
  for (let row = 0; row < 28; row += 1) {
    for (let col = 0; col < 28; col += 1) {
      const value = image[row]?.[col] ?? 0;
      weightedRow += row * value;
      weightedCol += col * value;
      mass += value;
      if (value > 0.08) {
        minRow = Math.min(minRow, row);
        minCol = Math.min(minCol, col);
        maxRow = Math.max(maxRow, row);
        maxCol = Math.max(maxCol, col);
      }
    }
  }

  const centerRow = mass > 1e-8 ? weightedRow / mass : 13.5;
  const centerCol = mass > 1e-8 ? weightedCol / mass : 13.5;
  const xOffset = (centerCol - 13.5) / 13.5;
  const yOffset = (centerRow - 13.5) / 13.5;

  const hasSignal = signalMask.some((value) => value > 0);
  const boxArea = hasSignal ? (maxRow - minRow + 1) * (maxCol - minCol + 1) : 0;
  const boxRatio = boxArea / 784;

  let symmetryPenalty = 0;
  let symmetryCount = 0;
  for (let row = 0; row < 28; row += 1) {
    for (let col = 0; col < 14; col += 1) {
      const left = image[row]?.[col] ?? 0;
      const right = image[row]?.[27 - col] ?? 0;
      symmetryPenalty += Math.abs(left - right);
      symmetryCount += 1;
    }
  }
  const symmetry = 1 - symmetryPenalty / Math.max(1, symmetryCount);

  return [
    clamp(density, 0, 1),
    clamp(xOffset, -1, 1),
    clamp(yOffset, -1, 1),
    clamp(boxRatio, 0, 1),
    clamp(symmetry, 0, 1),
  ];
};

const computeLearnPass = (
  weights: LearnWeights,
  input: number[],
  taskMode: LearnTaskMode,
  activationKind: LearnActivation,
  targetDigit: number,
  targetValue: number,
  l2: number
): LearnPassResult => {
  const z1 = Array.from({ length: HIDDEN_1 }, (_, hiddenIndex) => {
    let sum = weights.b1[hiddenIndex] ?? 0;
    for (let i = 0; i < INPUT_DIM; i += 1) sum += (input[i] ?? 0) * (weights.w1[i]?.[hiddenIndex] ?? 0);
    return sum;
  });
  const a1 = z1.map((value) => activation(value, activationKind));

  const z2 = Array.from({ length: HIDDEN_2 }, (_, hiddenIndex) => {
    let sum = weights.b2[hiddenIndex] ?? 0;
    for (let i = 0; i < HIDDEN_1; i += 1) sum += (a1[i] ?? 0) * (weights.w2[i]?.[hiddenIndex] ?? 0);
    return sum;
  });
  const a2 = z2.map((value) => activation(value, activationKind));

  const isClassification = taskMode === 'classification';
  const outputDim = isClassification ? CLASS_COUNT : 1;
  const z3 = Array.from({ length: outputDim }, (_, outIndex) => {
    let sum = isClassification ? (weights.b3c[outIndex] ?? 0) : (weights.b3r[0] ?? 0);
    for (let i = 0; i < HIDDEN_2; i += 1) {
      const w = isClassification ? (weights.w3c[i]?.[outIndex] ?? 0) : (weights.w3r[i]?.[0] ?? 0);
      sum += (a2[i] ?? 0) * w;
    }
    return sum;
  });

  const output = isClassification ? softmax(z3) : [z3[0]];
  const predictedClass = isClassification ? argMax(output) : Math.round(clamp(output[0] * 9, 0, 9));
  const safeTarget = clamp(Math.round(targetDigit), 0, 9);
  const safeValue = clamp(targetValue, 0, 1);

  const dz3 = Array.from({ length: outputDim }, (_, outIndex) => {
    if (isClassification) return output[outIndex] - (outIndex === safeTarget ? 1 : 0);
    return output[0] - safeValue;
  });

  const loss = isClassification
    ? -Math.log(Math.max(output[safeTarget] ?? 1e-10, 1e-10))
    : 0.5 * (output[0] - safeValue) * (output[0] - safeValue);

  const w3 = isClassification ? weights.w3c : weights.w3r;
  const dw3 = Array.from({ length: HIDDEN_2 }, (_, i) =>
    Array.from({ length: outputDim }, (_, j) => (a2[i] ?? 0) * (dz3[j] ?? 0) + l2 * (w3[i]?.[j] ?? 0))
  );
  const db3 = dz3.slice();

  const da2 = Array.from({ length: HIDDEN_2 }, (_, i) => {
    let sum = 0;
    for (let outIndex = 0; outIndex < outputDim; outIndex += 1) {
      sum += (dz3[outIndex] ?? 0) * (w3[i]?.[outIndex] ?? 0);
    }
    return sum;
  });

  const dz2 = Array.from(
    { length: HIDDEN_2 },
    (_, i) => da2[i] * activationPrime(z2[i], a2[i], activationKind)
  );
  const dw2 = Array.from({ length: HIDDEN_1 }, (_, i) =>
    Array.from({ length: HIDDEN_2 }, (_, j) => (a1[i] ?? 0) * (dz2[j] ?? 0) + l2 * (weights.w2[i]?.[j] ?? 0))
  );
  const db2 = dz2.slice();

  const da1 = Array.from({ length: HIDDEN_1 }, (_, i) => {
    let sum = 0;
    for (let j = 0; j < HIDDEN_2; j += 1) sum += (dz2[j] ?? 0) * (weights.w2[i]?.[j] ?? 0);
    return sum;
  });
  const dz1 = Array.from(
    { length: HIDDEN_1 },
    (_, i) => da1[i] * activationPrime(z1[i], a1[i], activationKind)
  );
  const dw1 = Array.from({ length: INPUT_DIM }, (_, i) =>
    Array.from({ length: HIDDEN_1 }, (_, j) => (input[i] ?? 0) * (dz1[j] ?? 0) + l2 * (weights.w1[i]?.[j] ?? 0))
  );
  const db1 = dz1.slice();

  return {
    z1,
    a1,
    z2,
    a2,
    z3,
    output,
    loss,
    predictedClass,
    gradients: {
      w1: dw1,
      b1: db1,
      w2: dw2,
      b2: db2,
      w3: dw3,
      b3: db3,
    },
  };
};

const applyGradientStep = (
  weights: LearnWeights,
  gradients: LearnGradients,
  taskMode: LearnTaskMode,
  learningRate: number
): LearnWeights => {
  const next: LearnWeights = {
    ...weights,
    w1: weights.w1.map((row, r) => row.map((value, c) => value - learningRate * (gradients.w1[r]?.[c] ?? 0))),
    b1: weights.b1.map((value, i) => value - learningRate * (gradients.b1[i] ?? 0)),
    w2: weights.w2.map((row, r) => row.map((value, c) => value - learningRate * (gradients.w2[r]?.[c] ?? 0))),
    b2: weights.b2.map((value, i) => value - learningRate * (gradients.b2[i] ?? 0)),
    w3c: weights.w3c.map((row) => row.slice()),
    b3c: weights.b3c.slice(),
    w3r: weights.w3r.map((row) => row.slice()),
    b3r: weights.b3r.slice(),
  };

  if (taskMode === 'classification') {
    next.w3c = weights.w3c.map((row, r) =>
      row.map((value, c) => value - learningRate * (gradients.w3[r]?.[c] ?? 0))
    );
    next.b3c = weights.b3c.map((value, i) => value - learningRate * (gradients.b3[i] ?? 0));
  } else {
    next.w3r = weights.w3r.map((row, r) => [
      row[0] - learningRate * (gradients.w3[r]?.[0] ?? 0),
    ]);
    next.b3r = [weights.b3r[0] - learningRate * (gradients.b3[0] ?? 0)];
  }

  return next;
};

function MiniMatrix({
  title,
  matrix,
  maxRows = 4,
  maxCols = 6,
}: {
  title: string;
  matrix: number[][];
  maxRows?: number;
  maxCols?: number;
}) {
  const rows = matrix.slice(0, maxRows);
  return (
    <article className="studio-nn-mini-matrix">
      <p className="studio-mini-label">{title}</p>
      <div className="studio-nn-mini-grid" style={{ gridTemplateColumns: `repeat(${Math.min(maxCols, matrix[0]?.length ?? 1)}, minmax(0, 1fr))` }}>
        {rows.map((row, r) =>
          row.slice(0, maxCols).map((value, c) => (
            <span key={`${title}-${r}-${c}`}>{value.toFixed(2)}</span>
          ))
        )}
      </div>
    </article>
  );
}

export function NeuralNetworkLearnLab({ image }: NeuralNetworkLearnLabProps) {
  const [taskMode, setTaskMode] = useState<LearnTaskMode>('classification');
  const [activationKind, setActivationKind] = useState<LearnActivation>('relu');
  const [targetDigit, setTargetDigit] = useState(0);
  const [targetValue, setTargetValue] = useState(0.5);
  const [learningRate, setLearningRate] = useState(0.08);
  const [l2, setL2] = useState(0.0005);
  const [stage, setStage] = useState(0);
  const [weights, setWeights] = useState<LearnWeights>(() => createInitialWeights());

  const hasSignal = useMemo(
    () => image.some((row) => row.some((value) => value > 0.06)),
    [image]
  );
  const features = useMemo(() => extractLearnFeatures(image), [image]);
  const pass = useMemo(
    () => computeLearnPass(weights, features, taskMode, activationKind, targetDigit, targetValue, l2),
    [activationKind, features, l2, targetDigit, targetValue, taskMode, weights]
  );

  const classTop3 = useMemo(() => {
    return pass.output
      .map((value, digit) => ({ digit, value }))
      .sort((left, right) => right.value - left.value)
      .slice(0, 3);
  }, [pass.output]);

  const gradientNorms = useMemo(
    () => ({
      dw1: l2NormMatrix(pass.gradients.w1),
      db1: l2NormVector(pass.gradients.b1),
      dw2: l2NormMatrix(pass.gradients.w2),
      db2: l2NormVector(pass.gradients.b2),
      dw3: l2NormMatrix(pass.gradients.w3),
      db3: l2NormVector(pass.gradients.b3),
    }),
    [pass.gradients]
  );

  const trainOneStep = () => {
    setWeights((previous) => {
      const innerPass = computeLearnPass(
        previous,
        features,
        taskMode,
        activationKind,
        targetDigit,
        targetValue,
        l2
      );
      return applyGradientStep(previous, innerPass.gradients, taskMode, learningRate);
    });
  };

  return (
    <section className="studio-card">
      <div className="studio-card-head">
        <h3>Neural Network Learn Lab (Forward + Backprop)</h3>
        <p>
          A compact educational network with matrix math: <strong>5 inputs → 5 hidden → 4 hidden → output</strong>.
          Task can switch between classification and regression.
        </p>
      </div>

      <div className="studio-nn-control-grid">
        <label className="deep-control">
          <span>Task</span>
          <select
            value={taskMode}
            onChange={(event) => setTaskMode(event.target.value as LearnTaskMode)}
          >
            <option value="classification">Classification (0-9)</option>
            <option value="regression">Regression (0.0-1.0)</option>
          </select>
        </label>
        <label className="deep-control">
          <span>Activation</span>
          <select
            value={activationKind}
            onChange={(event) => setActivationKind(event.target.value as LearnActivation)}
          >
            <option value="relu">ReLU</option>
            <option value="tanh">tanh</option>
          </select>
        </label>
        {taskMode === 'classification' ? (
          <label className="deep-control">
            <span>Target class</span>
            <select
              value={targetDigit}
              onChange={(event) => setTargetDigit(clamp(Number(event.target.value), 0, 9))}
            >
              {Array.from({ length: 10 }, (_, digit) => (
                <option key={`nn-target-${digit}`} value={digit}>{digit}</option>
              ))}
            </select>
          </label>
        ) : (
          <label className="deep-control">
            <span>Regression target</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={targetValue}
              onChange={(event) => setTargetValue(clamp(Number(event.target.value), 0, 1))}
            />
          </label>
        )}
        <label className="deep-control">
          <span>Learning rate</span>
          <input
            type="range"
            min="0.01"
            max="0.25"
            step="0.01"
            value={learningRate}
            onChange={(event) => setLearningRate(clamp(Number(event.target.value), 0.01, 0.25))}
          />
        </label>
        <label className="deep-control">
          <span>L2 regularization</span>
          <input
            type="range"
            min="0"
            max="0.01"
            step="0.0005"
            value={l2}
            onChange={(event) => setL2(clamp(Number(event.target.value), 0, 0.01))}
          />
        </label>
      </div>

      <div className="deep-step-row">
        {stageLabels.map((label, index) => (
          <button
            key={`nn-stage-${label}`}
            type="button"
            className={`studio-flow-node ${stage === index ? 'is-active' : ''}`}
            onClick={() => setStage(index)}
          >
            <span className="studio-flow-index">{index + 1}</span>
            <span className="studio-flow-label">{label}</span>
          </button>
        ))}
      </div>

      {!hasSignal && (
        <p className="studio-empty-copy">
          Draw a digit and press Predict. The learn lab then uses that image features for forward/backprop math.
        </p>
      )}

      {hasSignal && (
        <div className="studio-nn-learn-grid">
          <article className="studio-flow-panel studio-flow-animate">
            {stage === 0 && (
              <>
                <h4>Input Feature Vector</h4>
                <p><InlineMath math={'x\\in\\mathbb{R}^{1\\times5}'} /></p>
                <div className="studio-flow-bars">
                  {features.map((value, index) => (
                    <div key={`nn-feature-${featureLabels[index]}`} className="studio-flow-bar-row">
                      <span>{featureLabels[index]}</span>
                      <div className="studio-mini-bar"><span style={{ width: `${Math.max(2, Math.abs(value) * 100)}%` }} /></div>
                      <strong>{value.toFixed(3)}</strong>
                    </div>
                  ))}
                </div>
              </>
            )}

            {stage === 1 && (
              <>
                <h4>Hidden Layer 1 (5 units)</h4>
                <p><InlineMath math={'z_1=xW_1+b_1,\\quad a_1=f(z_1)'} /></p>
                <div className="studio-flow-bars">
                  {pass.a1.map((value, index) => (
                    <div key={`nn-a1-${index}`} className="studio-flow-bar-row">
                      <span>a1[{index}]</span>
                      <div className="studio-mini-bar"><span style={{ width: `${Math.max(2, Math.abs(value) * 100)}%` }} /></div>
                      <strong>{value.toFixed(3)}</strong>
                    </div>
                  ))}
                </div>
              </>
            )}

            {stage === 2 && (
              <>
                <h4>Hidden Layer 2 (4 units)</h4>
                <p><InlineMath math={'z_2=a_1W_2+b_2,\\quad a_2=f(z_2)'} /></p>
                <div className="studio-flow-bars">
                  {pass.a2.map((value, index) => (
                    <div key={`nn-a2-${index}`} className="studio-flow-bar-row">
                      <span>a2[{index}]</span>
                      <div className="studio-mini-bar"><span style={{ width: `${Math.max(2, Math.abs(value) * 100)}%` }} /></div>
                      <strong>{value.toFixed(3)}</strong>
                    </div>
                  ))}
                </div>
              </>
            )}

            {stage === 3 && (
              <>
                <h4>Output Layer</h4>
                {taskMode === 'classification' ? (
                  <>
                    <p><InlineMath math={'\\hat{y}=\\operatorname{softmax}(a_2W_3+b_3)'} /></p>
                    <div className="studio-flow-bars">
                      {classTop3.map((row) => (
                        <div key={`nn-top-${row.digit}`} className="studio-flow-bar-row">
                          <span>{row.digit}</span>
                          <div className="studio-mini-bar"><span style={{ width: `${Math.max(2, row.value * 100)}%` }} /></div>
                          <strong>{(row.value * 100).toFixed(1)}%</strong>
                        </div>
                      ))}
                    </div>
                    <p>Predicted class: <strong>{pass.predictedClass}</strong></p>
                  </>
                ) : (
                  <>
                    <p><InlineMath math={'\\hat{y}=a_2W_3+b_3'} /></p>
                    <p>Predicted value: <strong>{pass.output[0].toFixed(4)}</strong></p>
                  </>
                )}
              </>
            )}

            {stage === 4 && (
              <>
                <h4>Loss</h4>
                <p>
                  <InlineMath
                    math={
                      taskMode === 'classification'
                        ? `L=-\\log p(y=${targetDigit})=${pass.loss.toFixed(4)}`
                        : `L=\\frac{1}{2}(\\hat{y}-y)^2=${pass.loss.toFixed(4)}`
                    }
                  />
                </p>
                <p>
                  Hyperparameters: lr <strong>{learningRate.toFixed(2)}</strong>, L2 <strong>{l2.toFixed(4)}</strong>,
                  activation <strong>{activationKind}</strong>.
                </p>
              </>
            )}

            {stage === 5 && (
              <>
                <h4>Backpropagation</h4>
                <p><InlineMath math={'\\nabla W_3 \\rightarrow \\nabla W_2 \\rightarrow \\nabla W_1'} /></p>
                <div className="studio-flow-bars">
                  <div className="studio-flow-bar-row"><span>||dW1||</span><div className="studio-mini-bar"><span style={{ width: `${Math.min(100, gradientNorms.dw1 * 30)}%` }} /></div><strong>{gradientNorms.dw1.toFixed(3)}</strong></div>
                  <div className="studio-flow-bar-row"><span>||dW2||</span><div className="studio-mini-bar"><span style={{ width: `${Math.min(100, gradientNorms.dw2 * 30)}%` }} /></div><strong>{gradientNorms.dw2.toFixed(3)}</strong></div>
                  <div className="studio-flow-bar-row"><span>||dW3||</span><div className="studio-mini-bar"><span style={{ width: `${Math.min(100, gradientNorms.dw3 * 30)}%` }} /></div><strong>{gradientNorms.dw3.toFixed(3)}</strong></div>
                </div>
                <div className="deep-step-row">
                  <button type="button" className="deep-step-chip deep-step-chip-primary" onClick={trainOneStep}>
                    Train 1 Step
                  </button>
                  <button type="button" className="deep-step-chip" onClick={() => setWeights(createInitialWeights())}>
                    Reset Weights
                  </button>
                </div>
              </>
            )}
          </article>

          <div className="studio-nn-matrix-stack">
            <MiniMatrix title="W1 (5x5)" matrix={weights.w1} />
            <MiniMatrix title="W2 (5x4)" matrix={weights.w2} />
            <MiniMatrix title={taskMode === 'classification' ? 'W3 (4x10)' : 'W3 (4x1)'} matrix={taskMode === 'classification' ? weights.w3c : weights.w3r} />
          </div>
        </div>
      )}
    </section>
  );
}

