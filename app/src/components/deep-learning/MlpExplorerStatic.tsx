import { useMemo } from 'react';
import mlpStateSource from '@/data/deep-learning/mlp_states.json';
import narrationSource from '@/data/deep-learning/narration.json';
import type { DeepNarrationItem, DeepStepDelta, MlpStepMath } from '@/data/deep-learning/types';
import { DeepNarrationPanel } from './DeepNarrationPanel';
import { MlpMathPanel } from './MlpMathPanel';
import { MlpOutputPanel } from './MlpOutputPanel';

interface MlpExplorerStaticProps {
  hiddenLayers: number;
  neuronsPerLayer: number;
  activation: 'relu' | 'sigmoid';
  activeStep: number;
  showGuidedNarration: boolean;
}

type MlpStateSource = {
  input_features: string[];
  input_values: number[];
  base_hidden_weights: number[][];
  base_hidden_bias: number[];
  output_weights: number[][];
  output_bias: number[];
  activation_point?: Record<string, { z: number; a: number }>;
  output_confidence_gap?: Record<string, number>;
  step_deltas?: Record<string, DeepStepDelta[]>;
};

const mlpState = mlpStateSource as MlpStateSource;
const mlpNarration = narrationSource.mlp as DeepNarrationItem[];

const activate = (value: number, activationType: 'relu' | 'sigmoid') =>
  activationType === 'relu' ? Math.max(0, value) : 1 / (1 + Math.exp(-value));

const dot = (a: number[], b: number[]) => a.reduce((sum, value, idx) => sum + value * (b[idx] ?? 0), 0);

const softmax2 = (catLogit: number, dogLogit: number) => {
  const ea = Math.exp(catLogit);
  const eb = Math.exp(dogLogit);
  const total = ea + eb;
  return { cat: ea / total, dog: eb / total };
};

export function MlpExplorerStatic({
  hiddenLayers,
  neuronsPerLayer,
  activation,
  activeStep,
  showGuidedNarration,
}: MlpExplorerStaticProps) {
  const baseInput = mlpState.input_values;
  const hiddenNodeCount = Math.min(8, neuronsPerLayer);
  const hiddenVisualCount = Math.min(5, hiddenNodeCount);
  const layerCount = hiddenLayers + 2;
  const maxStep = layerCount - 2;
  const clampedStep = Math.max(0, Math.min(maxStep, activeStep));

  const network = useMemo(() => {
    const hiddenLayersValues: number[][] = [];
    const hiddenWeights: number[][][] = [];
    const hiddenBias: number[][] = [];

    for (let layerIdx = 0; layerIdx < hiddenLayers; layerIdx += 1) {
      const inputVector = layerIdx === 0 ? baseInput : hiddenLayersValues[layerIdx - 1];
      const weightsForLayer = Array.from({ length: hiddenNodeCount }).map((_, nodeIdx) =>
        inputVector.map((__, inputIdx) => {
          const seed = Math.sin((layerIdx + 1) * 1.33 + (nodeIdx + 1) * 0.71 + (inputIdx + 1) * 0.49);
          return Number((seed * 0.42).toFixed(3));
        })
      );
      const biasForLayer = Array.from({ length: hiddenNodeCount }).map((_, nodeIdx) =>
        Number((Math.cos((layerIdx + 1) * 0.53 + nodeIdx * 0.47) * 0.18).toFixed(3))
      );
      const outputs = weightsForLayer.map((weights, nodeIdx) =>
        activate(dot(inputVector, weights) + biasForLayer[nodeIdx], activation)
      );
      hiddenLayersValues.push(outputs.map((value) => Number(value.toFixed(4))));
      hiddenWeights.push(weightsForLayer);
      hiddenBias.push(biasForLayer);
    }

    const finalHidden = hiddenLayersValues[hiddenLayersValues.length - 1] ?? baseInput;
    const outputWeights = mlpState.output_weights.map((row) =>
      Array.from({ length: finalHidden.length }).map((__, idx) => row[idx % row.length])
    );
    const catLogit = dot(finalHidden, outputWeights[0]) + mlpState.output_bias[0];
    const dogLogit = dot(finalHidden, outputWeights[1]) + mlpState.output_bias[1];
    const probabilities = softmax2(catLogit, dogLogit);
    const confidenceGap = Math.abs(probabilities.cat - probabilities.dog);

    const nodeIndex = clampedStep % Math.max(1, hiddenNodeCount);
    const inputVector = clampedStep === 0 ? baseInput : hiddenLayersValues[clampedStep - 1];
    const weights = hiddenWeights[clampedStep]?.[nodeIndex]
      ?? Array.from({ length: inputVector.length }).map((__, idx) => mlpState.base_hidden_weights[nodeIndex % 4][idx % 4]);
    const bias = hiddenBias[clampedStep]?.[nodeIndex] ?? mlpState.base_hidden_bias[nodeIndex % 4];
    const z = dot(inputVector, weights) + bias;
    const activationOutput = activate(z, activation);

    const activationPoint = mlpState.activation_point?.[String(clampedStep)] ?? { z: Number(z.toFixed(4)), a: Number(activationOutput.toFixed(4)) };
    const outputConfidenceGap = mlpState.output_confidence_gap?.[String(clampedStep)] ?? Number(confidenceGap.toFixed(4));

    const stepMath: MlpStepMath = {
      inputVector: inputVector.map((value) => Number(value.toFixed(3))),
      weightVector: weights.map((value) => Number(value.toFixed(3))),
      bias: Number(bias.toFixed(3)),
      z: Number(z.toFixed(4)),
      activationOutput: Number(activationOutput.toFixed(4)),
      activationPoint: { z: Number(activationPoint.z.toFixed(4)), a: Number(activationPoint.a.toFixed(4)) },
      logits: [Number(catLogit.toFixed(4)), Number(dogLogit.toFixed(4))],
      probabilities: {
        cat: Number(probabilities.cat.toFixed(4)),
        dog: Number(probabilities.dog.toFixed(4)),
      },
      outputConfidenceGap: Number(outputConfidenceGap.toFixed(4)),
    };

    const prevLayerIdx = Math.max(0, clampedStep - 1);
    const prevNodeValues = hiddenLayersValues[prevLayerIdx] ?? [];
    const currentNodeValues = hiddenLayersValues[Math.min(hiddenLayersValues.length - 1, clampedStep)] ?? prevNodeValues;
    const prevAvg = prevNodeValues.length ? prevNodeValues.reduce((acc, value) => acc + value, 0) / prevNodeValues.length : 0;
    const currentAvg = currentNodeValues.length ? currentNodeValues.reduce((acc, value) => acc + value, 0) / currentNodeValues.length : activationOutput;

    const fallbackDeltas: DeepStepDelta[] = [
      {
        metric: 'z',
        previous: Number((z - 0.18).toFixed(4)),
        current: Number(z.toFixed(4)),
        delta: Number((0.18).toFixed(4)),
      },
      {
        metric: 'activation',
        previous: Number(prevAvg.toFixed(4)),
        current: Number(currentAvg.toFixed(4)),
        delta: Number((currentAvg - prevAvg).toFixed(4)),
      },
      {
        metric: 'confidence',
        previous: Number((confidenceGap - 0.07).toFixed(4)),
        current: Number(confidenceGap.toFixed(4)),
        delta: Number((0.07).toFixed(4)),
      },
    ];

    const keyedDeltas = mlpState.step_deltas?.[String(clampedStep)] ?? fallbackDeltas;
    const stepDeltas = keyedDeltas.map((item) => ({
      metric: item.metric,
      previous: Number(item.previous.toFixed(4)),
      current: Number(item.current.toFixed(4)),
      delta: Number(item.delta.toFixed(4)),
    }));

    return {
      stepMath,
      stepDeltas,
    };
  }, [activation, clampedStep, hiddenLayers, hiddenNodeCount]);

  const narration = mlpNarration[Math.min(mlpNarration.length - 1, clampedStep)] ?? null;

  const layerSizes = [
    mlpState.input_features.length,
    ...Array.from({ length: hiddenLayers }).map(() => hiddenVisualCount),
    2,
  ];
  const width = 680;
  const height = 230;
  const xGap = (width - 80) / Math.max(1, layerSizes.length - 1);

  const nodePositions = layerSizes.map((count, layerIdx) => {
    const yGap = count === 1 ? 0 : (height - 50) / Math.max(1, count - 1);
    return Array.from({ length: count }).map((__, nodeIdx) => ({
      x: 40 + layerIdx * xGap,
      y: count === 1 ? height / 2 : 28 + nodeIdx * yGap,
      layerIdx,
      nodeIdx,
    }));
  });

  return (
    <div className="deep-module-shell">
      <section className="deep-panel deep-panel-primary">
        <div className="deep-section-head">
          <h3 className="deep-section-title">MLP Studio</h3>
          <p className="deep-section-subtitle">Input features move through hidden layers to a cat/dog output head.</p>
          <p className="deep-progress-copy">Active step {clampedStep + 1} of {maxStep + 1}</p>
          <p className="deep-section-subtitle">
            Input ({mlpState.input_features.length}) {'->'} Hidden ({hiddenLayers} x {hiddenNodeCount}) {'->'} Output (2)
          </p>
        </div>
        <div className="deep-mlp-canvas">
          <svg viewBox={`0 0 ${width} ${height}`} className="deep-mlp-svg" role="img" aria-label="MLP network graph">
            {nodePositions.slice(0, -1).map((layerNodes, layerIdx) =>
              layerNodes.map((node) =>
                nodePositions[layerIdx + 1].map((nextNode) => (
                  <line
                    key={`${layerIdx}-${node.nodeIdx}-${nextNode.nodeIdx}`}
                    x1={node.x}
                    y1={node.y}
                    x2={nextNode.x}
                    y2={nextNode.y}
                    className={`deep-mlp-edge ${layerIdx === clampedStep ? 'deep-mlp-edge-active' : ''}`}
                  />
                ))
              )
            )}

            {nodePositions.map((layerNodes) =>
              layerNodes.map((node) => (
                <circle
                  key={`n-${node.layerIdx}-${node.nodeIdx}`}
                  cx={node.x}
                  cy={node.y}
                  r={6}
                  className={`deep-mlp-node ${activation === 'relu' ? 'deep-mlp-node-relu' : 'deep-mlp-node-sigmoid'} ${
                    node.layerIdx === clampedStep || node.layerIdx === clampedStep + 1 ? 'deep-mlp-node-active' : ''
                  }`}
                />
              ))
            )}

            <text x="12" y="16" className="deep-mlp-layer-title">Input (features)</text>
            <text x={Math.max(120, width / 2 - 42)} y="16" className="deep-mlp-layer-title">Hidden layer(s)</text>
            <text x={width - 110} y="16" className="deep-mlp-layer-title">Output (Cat/Dog)</text>
          </svg>
        </div>
      </section>

      <MlpMathPanel activation={activation} stepLabel={`Layer ${clampedStep + 1}`} math={network.stepMath} stepDeltas={network.stepDeltas} />
      <MlpOutputPanel math={network.stepMath} />
      {showGuidedNarration && (
        <DeepNarrationPanel
          title="Teaching Scaffold"
          narration={narration}
        />
      )}
    </div>
  );
}
