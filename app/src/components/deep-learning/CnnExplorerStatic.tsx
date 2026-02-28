import { useEffect, useMemo, useState } from 'react';
import pipelineSource from '@/data/deep-learning/cnn_pipeline.json';
import narrationSource from '@/data/deep-learning/narration.json';
import type { CnnPipelineStage, DeepNarrationItem, DeepResolvedImageRecord, DeepViewMode } from '@/data/deep-learning/types';
import { loadGrayscaleMatrix } from '@/lib/deepLearning/imageMatrix';
import { CnnPipelineDiagram } from './CnnPipelineDiagram';
import { CnnStageMathPanel } from './CnnStageMathPanel';
import { DeepNarrationPanel } from './DeepNarrationPanel';

interface CnnExplorerStaticProps {
  images: DeepResolvedImageRecord[];
  selectedImageId: number;
  selectedFilter: 'horizontal_edge' | 'vertical_edge';
  activeStageIndex: number;
  onStageIndexChange: (index: number) => void;
  showGuidedNarration: boolean;
  viewMode: DeepViewMode;
}

const stages = pipelineSource.stages as CnnPipelineStage[];
const narration = narrationSource.cnn as DeepNarrationItem[];

const horizontalKernel = [
  [1, 1, 1],
  [0, 0, 0],
  [-1, -1, -1],
];
const verticalKernel = [
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1],
];

const relu = (value: number) => Math.max(0, value);
const dot = (a: number[][], b: number[][]) =>
  a.reduce((sum, row, r) => sum + row.reduce((rowSum, value, c) => rowSum + value * b[r][c], 0), 0);

const softmax2 = (a: number, b: number) => {
  const ea = Math.exp(a);
  const eb = Math.exp(b);
  const total = ea + eb;
  return { cat: ea / total, dog: eb / total };
};

export function CnnExplorerStatic({
  images,
  selectedImageId,
  selectedFilter,
  activeStageIndex,
  onStageIndexChange,
  showGuidedNarration,
  viewMode,
}: CnnExplorerStaticProps) {
  const [inputMatrix, setInputMatrix] = useState<number[][]>(Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => 0)));

  const selectedImage = images.find((image) => image.id === selectedImageId) ?? images[0];

  useEffect(() => {
    let mounted = true;
    loadGrayscaleMatrix(selectedImage.src, 8)
      .then((matrix) => {
        if (mounted) setInputMatrix(matrix);
      })
      .catch(() => {
        if (mounted) {
          setInputMatrix(Array.from({ length: 8 }, (_, r) =>
            Array.from({ length: 8 }, (_, c) => Number((((r + c) % 6) / 5).toFixed(2)))
          ));
        }
      });
    return () => {
      mounted = false;
    };
  }, [selectedImage.src]);

  const kernel = selectedFilter === 'horizontal_edge' ? horizontalKernel : verticalKernel;

  const convMatrix = useMemo(() => {
    const rows = inputMatrix.length - 2;
    const cols = inputMatrix[0].length - 2;
    return Array.from({ length: rows }).map((_, r) =>
      Array.from({ length: cols }).map((__, c) => {
        const patch = Array.from({ length: 3 }).map((___, pr) =>
          Array.from({ length: 3 }).map((____, pc) => inputMatrix[r + pr][c + pc])
        );
        return Number(dot(patch, kernel).toFixed(3));
      })
    );
  }, [inputMatrix, kernel]);

  const reluMatrix = useMemo(() => convMatrix.map((row) => row.map((value) => Number(relu(value).toFixed(3)))), [convMatrix]);

  const poolMatrix = useMemo(() => {
    const rows = Math.floor(reluMatrix.length / 2);
    const cols = Math.floor(reluMatrix[0].length / 2);
    return Array.from({ length: rows }).map((_, r) =>
      Array.from({ length: cols }).map((__, c) => {
        const values = [
          reluMatrix[r * 2][c * 2],
          reluMatrix[r * 2][c * 2 + 1],
          reluMatrix[r * 2 + 1][c * 2],
          reluMatrix[r * 2 + 1][c * 2 + 1],
        ];
        return Number(Math.max(...values).toFixed(3));
      })
    );
  }, [reluMatrix]);

  const flattened = poolMatrix.flat();
  const catLogit = flattened.reduce((sum, value, idx) => sum + value * (idx % 2 === 0 ? 0.42 : 0.19), 0) + (selectedImage.class === 'cat' ? 0.5 : 0.18);
  const dogLogit = flattened.reduce((sum, value, idx) => sum + value * (idx % 2 === 0 ? 0.17 : 0.37), 0) + (selectedImage.class === 'dog' ? 0.5 : 0.18);
  const probabilities = softmax2(catLogit, dogLogit);

  const convRows = Math.max(1, convMatrix.length);
  const convCols = Math.max(1, convMatrix[0]?.length ?? 1);
  const clampedStageIndex = Math.max(0, Math.min(stages.length - 1, activeStageIndex));
  const convIndex = Math.min(Math.max(clampedStageIndex * 2, 0), convRows * convCols - 1);
  const convRow = Math.floor(convIndex / convCols) % convRows;
  const convCol = convIndex % convCols;

  const currentStage = stages[clampedStageIndex] ?? stages[0];
  const currentNarration = narration.find((item) => item.step_id === currentStage.id) ?? narration[narration.length - 1] ?? null;

  return (
    <div className="deep-module-shell">
      <section className="deep-panel deep-panel-primary">
        <div className="deep-section-head">
          <h3 className="deep-section-title">CNN Studio</h3>
          <p className="deep-section-subtitle">Input {'->'} Conv {'->'} ReLU {'->'} Pool {'->'} Flatten {'->'} Dense</p>
          <p className="deep-progress-copy">Active stage {activeStageIndex + 1} of {stages.length}: {currentStage.label}</p>
        </div>
        <p className="deep-section-subtitle">Sample and filter controls are in the inspector sidebar for stable stage flow.</p>
      </section>

      <CnnPipelineDiagram
        stages={stages}
        activeStageId={currentStage.id}
        onStageChange={(id) => {
          const idx = stages.findIndex((stage) => stage.id === id);
          if (idx >= 0 && idx !== clampedStageIndex) onStageIndexChange(idx);
        }}
      />

      <CnnStageMathPanel
        stage={currentStage}
        viewMode={viewMode}
        kernel={kernel}
        inputMatrix={inputMatrix}
        convMatrix={convMatrix}
        reluMatrix={reluMatrix}
        poolMatrix={poolMatrix}
        flattened={flattened}
        probabilities={probabilities}
        convRow={convRow}
        convCol={convCol}
      />

      {showGuidedNarration && (
        <DeepNarrationPanel
          title="Teaching Scaffold"
          narration={currentNarration}
        />
      )}
    </div>
  );
}
