import { useEffect, useMemo, useState } from 'react';
import { scaleLinear } from 'd3-scale';
import type { DlInferenceMode } from '@/data/deep-learning/types';
import { runMlpInference } from '@/lib/deepLearning/mlpTrace';
import type { MnistMlpModel } from '@/lib/deepLearning/modelLoader';
import {
  centerImageByMass,
  flattenImageMatrix,
  groupInto16,
  topKIndices,
} from '@/lib/deepLearning/mnistPreprocess';

interface MlpInferenceExplorerProps {
  image: number[][];
  model: MnistMlpModel | null;
  predictNonce: number;
  inferenceMode: DlInferenceMode;
}

interface NodePosition {
  x: number;
  y: number;
}

interface InputTerm {
  groupIndex: number;
  inputValue: number;
  effectiveWeight: number;
  contribution: number;
}

interface HiddenNodeDetail {
  hiddenIndex: number;
  bias: number;
  z: number;
  activation: number;
  terms: InputTerm[];
  outputWeights: number[];
  outputContributions: number[];
  predictedContribution: number;
}

interface IncomingEdge {
  hiddenIndex: number;
  groupIndex: number;
  weight: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface OutgoingEdge {
  hiddenIndex: number;
  outputIndex: number;
  weight: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface GhostPath {
  hiddenIndex: number;
  groupIndex: number;
  outputIndex: number;
  incomingWeight: number;
  outgoingWeight: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  x3: number;
  y3: number;
}

const SVG_WIDTH = 820;
const SVG_HEIGHT = 460;
const VISIBLE_HIDDEN_COUNT = 12;
const groupIndexMap = buildGroupIndexMap();

function buildGroupIndexMap(): number[][] {
  const groups = Array.from({ length: 16 }, () => [] as number[]);
  for (let row = 0; row < 28; row += 1) {
    for (let col = 0; col < 28; col += 1) {
      const groupRow = Math.min(3, Math.floor(row / 7));
      const groupCol = Math.min(3, Math.floor(col / 7));
      groups[groupRow * 4 + groupCol].push(row * 28 + col);
    }
  }
  return groups;
}

export function MlpInferenceExplorer({
  image,
  model,
  predictNonce,
  inferenceMode,
}: MlpInferenceExplorerProps) {
  const [hoveredHidden, setHoveredHidden] = useState<number | null>(null);
  const [pinnedHidden, setPinnedHidden] = useState<number | null>(null);
  const hasSignal = useMemo(
    () => image.some((row) => row.some((value) => value > 0.06)),
    [image]
  );

  const analysis = useMemo(() => {
    if (!model || !hasSignal) return null;

    const centered = centerImageByMass(image);
    const inputVector = flattenImageMatrix(centered);
    const inputGroups = groupInto16(centered);
    const inference = runMlpInference(centered, model, {
      topK: VISIBLE_HIDDEN_COUNT,
      hiddenCap: VISIBLE_HIDDEN_COUNT,
      groupEdgesPerHidden: 2,
      inferenceMode,
    });

    const predictedClass = inference.snapshot.predictedClass;
    const hiddenScores = inference.trace.hiddenActivations.map(
      (activation, hiddenIndex) =>
        activation * (model.w2[predictedClass]?.[hiddenIndex] ?? 0)
    );
    const visibleHiddenIndices = topKIndices(hiddenScores, VISIBLE_HIDDEN_COUNT);

    const hiddenNodes = visibleHiddenIndices
      .map((hiddenIndex): HiddenNodeDetail => {
        const terms = groupIndexMap.map((pixelIndices, groupIndex) => {
          let contribution = 0;
          for (let i = 0; i < pixelIndices.length; i += 1) {
            const pixelIndex = pixelIndices[i];
            contribution +=
              (inputVector[pixelIndex] ?? 0) *
              (model.w1[hiddenIndex]?.[pixelIndex] ?? 0);
          }
          const inputValue = inputGroups[groupIndex] ?? 0;
          const effectiveWeight =
            inputValue > 1e-8 ? contribution / inputValue : 0;
          return {
            groupIndex,
            inputValue,
            effectiveWeight,
            contribution,
          };
        });

        const bias = model.b1[hiddenIndex] ?? 0;
        const weightedSum = terms.reduce((sum, term) => sum + term.contribution, 0);
        const z = weightedSum + bias;
        const activation = z > 0 ? z : 0;
        const outputWeights = Array.from(
          { length: model.outputSize },
          (_, outputIndex) => model.w2[outputIndex]?.[hiddenIndex] ?? 0
        );
        const outputContributions = outputWeights.map(
          (weight) => weight * activation
        );

        return {
          hiddenIndex,
          bias,
          z,
          activation,
          terms,
          outputWeights,
          outputContributions,
          predictedContribution: outputContributions[predictedClass] ?? 0,
        };
      })
      .sort(
        (left, right) =>
          Math.abs(right.predictedContribution) - Math.abs(left.predictedContribution)
      );

    return {
      snapshot: inference.snapshot,
      predictedClass,
      inputGroups,
      hiddenNodes,
    };
  }, [hasSignal, image, inferenceMode, model]);

  useEffect(() => {
    setHoveredHidden(null);
    setPinnedHidden(null);
  }, [predictNonce]);

  const activeHidden = pinnedHidden ?? hoveredHidden;
  const activeNode = useMemo(
    () =>
      analysis?.hiddenNodes.find((node) => node.hiddenIndex === activeHidden) ??
      null,
    [activeHidden, analysis]
  );

  const contributorSummary = useMemo(() => {
    if (!activeNode) return null;

    const roundedZeroCutoff = 0.0005;
    const filtered = activeNode.terms
      .filter(
        (term) =>
          Math.abs(term.inputValue) >= roundedZeroCutoff &&
          Math.abs(term.effectiveWeight) >= roundedZeroCutoff
      )
      .sort(
        (left, right) =>
          Math.abs(right.contribution) - Math.abs(left.contribution)
      );

    const positiveTop = filtered
      .filter((term) => term.contribution > 0)
      .slice(0, 3);
    const negativeTop = filtered
      .filter((term) => term.contribution < 0)
      .slice(0, 3);

    const positivePull = filtered.reduce(
      (sum, term) => sum + Math.max(0, term.contribution),
      0
    );
    const negativePull = filtered.reduce(
      (sum, term) => sum + Math.min(0, term.contribution),
      0
    );
    const pullScale = positivePull + Math.abs(negativePull);

    return {
      filteredCount: filtered.length,
      positiveTop,
      negativeTop,
      positivePull,
      negativePull,
      positiveWidth: pullScale > 0 ? (positivePull / pullScale) * 100 : 50,
      negativeWidth:
        pullScale > 0 ? (Math.abs(negativePull) / pullScale) * 100 : 50,
    };
  }, [activeNode]);

  const inputPositions = useMemo(() => {
    return Array.from({ length: 16 }, (_, index) => {
      const row = Math.floor(index / 4);
      const col = index % 4;
      return {
        x: 86 + col * 54,
        y: 86 + row * 72,
      } satisfies NodePosition;
    });
  }, []);

  const hiddenPositions = useMemo(() => {
    const positions = new Map<number, NodePosition>();
    if (!analysis) return positions;
    const minY = 58;
    const maxY = 402;
    const denominator = Math.max(1, analysis.hiddenNodes.length - 1);
    for (let i = 0; i < analysis.hiddenNodes.length; i += 1) {
      positions.set(analysis.hiddenNodes[i].hiddenIndex, {
        x: 410,
        y: minY + ((maxY - minY) * i) / denominator,
      });
    }
    return positions;
  }, [analysis]);

  const outputPositions = useMemo(() => {
    return Array.from({ length: 10 }, (_, outputIndex) => ({
      x: 730,
      y: 50 + outputIndex * 40,
    }));
  }, []);

  const incomingEdges = useMemo(() => {
    if (!analysis) return [] as IncomingEdge[];
    const edges: IncomingEdge[] = [];
    for (let i = 0; i < analysis.hiddenNodes.length; i += 1) {
      const node = analysis.hiddenNodes[i];
      const hiddenPosition = hiddenPositions.get(node.hiddenIndex);
      if (!hiddenPosition) continue;
      for (let j = 0; j < node.terms.length; j += 1) {
        const term = node.terms[j];
        edges.push({
          hiddenIndex: node.hiddenIndex,
          groupIndex: term.groupIndex,
          weight: term.effectiveWeight,
          x1: inputPositions[term.groupIndex].x,
          y1: inputPositions[term.groupIndex].y,
          x2: hiddenPosition.x,
          y2: hiddenPosition.y,
        });
      }
    }
    return edges;
  }, [analysis, hiddenPositions, inputPositions]);

  const outgoingEdges = useMemo(() => {
    if (!analysis) return [] as OutgoingEdge[];
    const edges: OutgoingEdge[] = [];
    for (let i = 0; i < analysis.hiddenNodes.length; i += 1) {
      const node = analysis.hiddenNodes[i];
      const hiddenPosition = hiddenPositions.get(node.hiddenIndex);
      if (!hiddenPosition) continue;
      for (let outputIndex = 0; outputIndex < node.outputWeights.length; outputIndex += 1) {
        edges.push({
          hiddenIndex: node.hiddenIndex,
          outputIndex,
          weight: node.outputWeights[outputIndex],
          x1: hiddenPosition.x,
          y1: hiddenPosition.y,
          x2: outputPositions[outputIndex].x,
          y2: outputPositions[outputIndex].y,
        });
      }
    }
    return edges;
  }, [analysis, hiddenPositions, outputPositions]);

  const incomingWeightMax = useMemo(() => {
    if (incomingEdges.length === 0) return 1;
    return Math.max(...incomingEdges.map((edge) => Math.abs(edge.weight)), 1e-6);
  }, [incomingEdges]);

  const outgoingWeightMax = useMemo(() => {
    if (outgoingEdges.length === 0) return 1;
    return Math.max(...outgoingEdges.map((edge) => Math.abs(edge.weight)), 1e-6);
  }, [outgoingEdges]);

  const incomingWidthScale = useMemo(
    () =>
      scaleLinear<number, number>()
        .domain([0, incomingWeightMax])
        .range([0.8, 4.8])
        .clamp(true),
    [incomingWeightMax]
  );

  const outgoingWidthScale = useMemo(
    () =>
      scaleLinear<number, number>()
        .domain([0, outgoingWeightMax])
        .range([0.8, 4.6])
        .clamp(true),
    [outgoingWeightMax]
  );

  const ghostPaths = useMemo(() => {
    if (!analysis) return [] as GhostPath[];
    const candidates: Array<GhostPath & { score: number }> = [];

    for (let i = 0; i < analysis.hiddenNodes.length; i += 1) {
      const node = analysis.hiddenNodes[i];
      const hiddenPosition = hiddenPositions.get(node.hiddenIndex);
      if (!hiddenPosition) continue;

      for (let groupIndex = 0; groupIndex < node.terms.length; groupIndex += 1) {
        const term = node.terms[groupIndex];
        const incomingWeight = term.effectiveWeight;
        if (Math.abs(incomingWeight) < 1e-6) continue;

        for (let outputIndex = 0; outputIndex < node.outputWeights.length; outputIndex += 1) {
          const outgoingWeight = node.outputWeights[outputIndex] ?? 0;
          const score = Math.abs(incomingWeight * outgoingWeight);
          if (score < 1e-6) continue;
          candidates.push({
            hiddenIndex: node.hiddenIndex,
            groupIndex,
            outputIndex,
            incomingWeight,
            outgoingWeight,
            x1: inputPositions[groupIndex].x,
            y1: inputPositions[groupIndex].y,
            x2: hiddenPosition.x,
            y2: hiddenPosition.y,
            x3: outputPositions[outputIndex].x,
            y3: outputPositions[outputIndex].y,
            score,
          });
        }
      }
    }

    return candidates
      .sort((left, right) => right.score - left.score)
      .slice(0, 5)
      .map(({ score: _score, ...path }) => path);
  }, [analysis, hiddenPositions, inputPositions, outputPositions]);

  if (!analysis) {
    return (
      <section className="studio-mlp-shell">
        <div className="studio-network-empty">
          Draw a digit and click <strong>Predict</strong> to generate the MLP map.
        </div>
      </section>
    );
  }

  const isPinned = pinnedHidden !== null;
  const weightedSum = activeNode
    ? activeNode.terms.reduce((sum, term) => sum + term.contribution, 0)
    : 0;

  return (
    <section className="studio-mlp-shell">
      <div className="studio-network-head">
        <h4>Node-Link View (D3-scaled)</h4>
        <p>
          Hidden nodes are visible first. Hover a hidden node to reveal only its
          incoming and outgoing weights.
        </p>
      </div>

      <div className="studio-network-stage">
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="studio-network-svg"
          role="img"
          aria-label="MLP network with hover-reveal edges"
        >
          <text x="54" y="28" className="studio-layer-title">
            Input Groups
          </text>
          <text x="366" y="28" className="studio-layer-title">
            Hidden Layer
          </text>
          <text x="694" y="28" className="studio-layer-title">
            Output (0-9)
          </text>

          {ghostPaths.map((path) => (
            <g key={`ghost-${path.hiddenIndex}-${path.groupIndex}-${path.outputIndex}`}>
              <line
                x1={path.x1}
                y1={path.y1}
                x2={path.x2}
                y2={path.y2}
                className="studio-edge-ghost"
                style={{ opacity: activeHidden === null ? 0.05 : 0 }}
                stroke={
                  path.incomingWeight >= 0
                    ? 'var(--studio-signal-positive)'
                    : 'var(--studio-signal-negative)'
                }
                strokeWidth={incomingWidthScale(Math.abs(path.incomingWeight))}
              />
              <line
                x1={path.x2}
                y1={path.y2}
                x2={path.x3}
                y2={path.y3}
                className="studio-edge-ghost"
                style={{ opacity: activeHidden === null ? 0.05 : 0 }}
                stroke={
                  path.outgoingWeight >= 0
                    ? 'var(--studio-signal-positive)'
                    : 'var(--studio-signal-negative)'
                }
                strokeWidth={outgoingWidthScale(Math.abs(path.outgoingWeight))}
              />
            </g>
          ))}

          {incomingEdges.map((edge) => (
            <line
              key={`incoming-${edge.hiddenIndex}-${edge.groupIndex}`}
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              className={`studio-edge ${
                activeHidden === edge.hiddenIndex ? 'is-visible' : ''
              }`}
              stroke={
                edge.weight >= 0
                  ? 'var(--studio-signal-positive)'
                  : 'var(--studio-signal-negative)'
              }
              strokeWidth={incomingWidthScale(Math.abs(edge.weight))}
            />
          ))}

          {outgoingEdges.map((edge) => (
            <line
              key={`outgoing-${edge.hiddenIndex}-${edge.outputIndex}`}
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              className={`studio-edge ${
                activeHidden === edge.hiddenIndex ? 'is-visible' : ''
              }`}
              stroke={
                edge.weight >= 0
                  ? 'var(--studio-signal-positive)'
                  : 'var(--studio-signal-negative)'
              }
              strokeWidth={outgoingWidthScale(Math.abs(edge.weight))}
            />
          ))}

          {analysis.inputGroups.map((value, index) => (
            <g key={`input-node-${index}`}>
              <circle
                cx={inputPositions[index].x}
                cy={inputPositions[index].y}
                r={10}
                className="studio-node studio-node-input"
                style={{
                  opacity: `${0.45 + value * 0.55}`,
                }}
              >
                <title>{`Input group ${index + 1}\nActivation: ${value.toFixed(4)}`}</title>
              </circle>
              <text
                x={inputPositions[index].x}
                y={inputPositions[index].y + 4}
                className="studio-node-label"
              >
                {index + 1}
              </text>
            </g>
          ))}

          {analysis.hiddenNodes.map((node) => {
            const position = hiddenPositions.get(node.hiddenIndex);
            if (!position) return null;
            const selected = node.hiddenIndex === activeHidden;
            return (
              <g
                key={`hidden-node-${node.hiddenIndex}`}
                role="button"
                tabIndex={0}
                onMouseEnter={() => setHoveredHidden(node.hiddenIndex)}
                onMouseLeave={() => setHoveredHidden(null)}
                onClick={() =>
                  setPinnedHidden((previous) =>
                    previous === node.hiddenIndex ? null : node.hiddenIndex
                  )
                }
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  setPinnedHidden((previous) =>
                    previous === node.hiddenIndex ? null : node.hiddenIndex
                  );
                }}
              >
                <circle
                  cx={position.x}
                  cy={position.y}
                  r={selected ? 11.5 : 9.5}
                  className={`studio-node studio-node-hidden ${
                    selected ? 'is-active' : ''
                  }`}
                >
                  <title>{`h${node.hiddenIndex}\nz=${node.z.toFixed(4)}\na=${node.activation.toFixed(4)}\nClick to pin`}</title>
                </circle>
              </g>
            );
          })}

          {analysis.snapshot.probabilities.map((probability, outputIndex) => (
            <g key={`output-node-${outputIndex}`}>
              <circle
                cx={outputPositions[outputIndex].x}
                cy={outputPositions[outputIndex].y}
                r={outputIndex === analysis.predictedClass ? 11 : 9}
                className={`studio-node studio-node-output ${
                  outputIndex === analysis.predictedClass ? 'is-active' : ''
                }`}
                style={{ opacity: `${0.5 + probability * 0.5}` }}
              >
                <title>{`Class ${outputIndex}\nProbability ${(probability * 100).toFixed(2)}%`}</title>
              </circle>
              <text
                x={outputPositions[outputIndex].x + 16}
                y={outputPositions[outputIndex].y + 4}
                className="studio-node-label studio-output-index"
              >
                {outputIndex}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <article className="studio-math-card">
        <div className="studio-math-head">
          <h5>Dynamic Math Card</h5>
          <p>
            {activeNode
              ? `Node h${activeNode.hiddenIndex} is active`
              : 'Hover a hidden node to reveal exact node math'}
          </p>
          {isPinned && (
            <button
              type="button"
              className="studio-clear-pin"
              onClick={() => setPinnedHidden(null)}
            >
              Clear pinned node
            </button>
          )}
        </div>

        {!activeNode && (
          <p className="studio-math-empty">
            The equation and edge values appear only on interaction to avoid
            overload.
          </p>
        )}

        {activeNode && (
          <div className="studio-math-grid">
            <div className="studio-equation-panel">
              <p className="studio-equation-template">
                z = (x1*w1) + (x2*w2) + ... + (x16*w16) + b
              </p>
              <div className="studio-term-columns">
                <div className="studio-term-group">
                  <p className="studio-term-head is-positive">Top + contributors</p>
                  {contributorSummary?.positiveTop.length ? (
                    contributorSummary.positiveTop.map((term) => (
                      <span
                        key={`term-pos-${term.groupIndex}`}
                        className="studio-term is-positive"
                      >
                        x{term.groupIndex + 1} {term.inputValue.toFixed(3)} * w
                        {term.groupIndex + 1} {term.effectiveWeight.toFixed(3)} ={' '}
                        {term.contribution.toFixed(3)}
                      </span>
                    ))
                  ) : (
                    <p className="studio-term-empty">No positive contributors.</p>
                  )}
                </div>
                <div className="studio-term-group">
                  <p className="studio-term-head is-negative">Top - contributors</p>
                  {contributorSummary?.negativeTop.length ? (
                    contributorSummary.negativeTop.map((term) => (
                      <span
                        key={`term-neg-${term.groupIndex}`}
                        className="studio-term is-negative"
                      >
                        x{term.groupIndex + 1} {term.inputValue.toFixed(3)} * w
                        {term.groupIndex + 1} {term.effectiveWeight.toFixed(3)} ={' '}
                        {term.contribution.toFixed(3)}
                      </span>
                    ))
                  ) : (
                    <p className="studio-term-empty">No negative contributors.</p>
                  )}
                </div>
              </div>
              <p className="studio-term-footnote">
                Showing {contributorSummary?.filteredCount ?? 0} non-zero terms
                (input and weight both above 0.000 after rounding).
              </p>
              <div className="studio-z-bar-shell" aria-label="Positive and negative pull toward z">
                <div className="studio-z-bar-track">
                  <span
                    className="is-negative"
                    style={{ width: `${contributorSummary?.negativeWidth ?? 50}%` }}
                  />
                  <span
                    className="is-positive"
                    style={{ width: `${contributorSummary?.positiveWidth ?? 50}%` }}
                  />
                </div>
                <p className="studio-z-bar-copy">
                  Pull = +{(contributorSummary?.positivePull ?? 0).toFixed(3)} + (
                  {(contributorSummary?.negativePull ?? 0).toFixed(3)}) + b (
                  {activeNode.bias.toFixed(3)}) = z ({activeNode.z.toFixed(3)})
                </p>
              </div>
              <p className="studio-equation-result">
                Sum = {weightedSum.toFixed(3)} | b = {activeNode.bias.toFixed(3)} | z
                = {activeNode.z.toFixed(3)} | ReLU(z) ={' '}
                {activeNode.activation.toFixed(3)}
              </p>
            </div>

            <div className="studio-output-panel">
              <p className="studio-mini-label">Class Probabilities</p>
              <div className="studio-output-bars">
                {analysis.snapshot.probabilities.map((value, digit) => (
                  <div key={`output-bar-${digit}`} className="studio-output-row">
                    <span>{digit}</span>
                    <div className="studio-output-track">
                      <span
                        className={
                          digit === analysis.predictedClass ? 'is-active' : ''
                        }
                        style={{ width: `${Math.max(2, value * 100)}%` }}
                      />
                    </div>
                    <strong>{(value * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
              <p className="studio-node-impact">
                Contribution to predicted class {analysis.predictedClass}:{' '}
                <strong>{activeNode.predictedContribution.toFixed(3)}</strong>
              </p>
            </div>
          </div>
        )}
      </article>
    </section>
  );
}
