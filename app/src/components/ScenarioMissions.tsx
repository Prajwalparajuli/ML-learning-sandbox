import { useMemo, useState } from 'react';
import { Lock, Sparkles } from 'lucide-react';
import { useModelStore } from '../store/modelStore';

type ActiveMission = {
  id: string;
  title: string;
  prompt: string;
  expected: string;
  apply: () => void;
};

type PlannedMission = {
  id: string;
  title: string;
  prompt: string;
  expected: string;
  phase: string;
};

interface ScenarioMissionsProps {
  compact?: boolean;
}

export function ScenarioMissions({ compact = false }: ScenarioMissionsProps) {
  const [pulseMission, setPulseMission] = useState<string | null>(null);
  const [showAllCompact, setShowAllCompact] = useState(false);
  const store = useModelStore();
  const taskMode = store.taskMode;

  const regressionMissions: ActiveMission[] = useMemo(
    () => [
      {
        id: 'fix-overfit',
        title: 'Fix Overfit',
        prompt: 'Start from a too-flexible setup and stabilize validation behavior.',
        expected: 'Validation error curve should flatten and metric volatility should drop.',
        apply: () => {
          store.setFeatureMode('1d');
          store.setDataset('noisy');
          store.setModelType('polynomial');
          store.setParam('polynomialDegree', 6);
          store.setEvaluationMode('cross_validation');
          store.setCvFolds(5);
          store.setShowAssumptions(true);
          store.setCompareWithOls(false);
          store.regenerateDataset();
        },
      },
      {
        id: 'recover-underfit',
        title: 'Recover Underfit',
        prompt: 'Move from a simple line to a curve-aware model on nonlinear data.',
        expected: 'Fit line should follow curvature and RMSE should trend down.',
        apply: () => {
          store.setFeatureMode('1d');
          store.setDataset('quadratic');
          store.setModelType('polynomial');
          store.setParam('polynomialDegree', 3);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.25);
          store.setShowAssumptions(false);
          store.setCompareWithOls(true);
          store.regenerateDataset();
        },
      },
      {
        id: 'outlier-stress',
        title: 'Outlier Stress Test',
        prompt: 'Stress assumptions and compare regularization against OLS.',
        expected: 'Residual spread should reveal instability in naive fits.',
        apply: () => {
          store.setFeatureMode('1d');
          store.setDataset('outliers');
          store.setModelType('ridge');
          store.setParam('alpha', 0.45);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.3);
          store.setShowAssumptions(true);
          store.setCompareWithOls(true);
          store.regenerateDataset();
        },
      },
      {
        id: 'stabilize-ridge',
        title: 'Stabilize with Ridge',
        prompt: 'Use shrinkage to make noisy gradients more stable.',
        expected: 'Bias rises slightly while generalization consistency improves.',
        apply: () => {
          store.setFeatureMode('1d');
          store.setDataset('heteroscedastic');
          store.setModelType('ridge');
          store.setParam('alpha', 0.9);
          store.setEvaluationMode('cross_validation');
          store.setCvFolds(6);
          store.setShowAssumptions(true);
          store.setCompareWithOls(true);
          store.regenerateDataset();
        },
      },
    ],
    [store]
  );
  const classificationMissions: ActiveMission[] = useMemo(
    () => [
      {
        id: 'cls-boundary',
        title: 'Tune Boundary',
        prompt: 'Compare linear and nonlinear separation behavior.',
        expected: 'Boundary shape changes while confusion counts shift.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_overlap');
          store.setModelType('svm_classifier');
          store.setParam('svmC', 1.3);
          store.setParam('decisionThreshold', 0.5);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.25);
          store.regenerateDataset();
        },
      },
      {
        id: 'cls-imbalance',
        title: 'Imbalance Tradeoff',
        prompt: 'Improve recall on rare positives without collapsing precision.',
        expected: 'Threshold and model choice change minority-class recovery.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_imbalanced');
          store.setModelType('logistic_classifier');
          store.setParam('decisionThreshold', 0.35);
          store.setEvaluationMode('cross_validation');
          store.setCvFolds(5);
          store.regenerateDataset();
        },
      },
      {
        id: 'cls-threshold',
        title: 'Threshold Sweep',
        prompt: 'Raise and lower threshold to observe precision/recall exchange.',
        expected: 'Higher threshold usually lowers false positives but misses more positives.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_linear');
          store.setModelType('logistic_classifier');
          store.setParam('decisionThreshold', 0.7);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.25);
          store.regenerateDataset();
        },
      },
      {
        id: 'cls-knn',
        title: 'Local vs Global',
        prompt: 'Contrast KNN locality with logistic global separation.',
        expected: 'KNN adapts to local pockets; logistic keeps one smooth boundary.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_moons');
          store.setModelType('knn_classifier');
          store.setParam('knnK', 7);
          store.setParam('decisionThreshold', 0.5);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.3);
          store.regenerateDataset();
        },
      },
      {
        id: 'cls-tree-split',
        title: 'Tree Split Explorer',
        prompt: 'Compare greedy partitioning against smoother separators on overlap.',
        expected: 'Sharper local splits can improve fit but increase variance sensitivity.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_overlap');
          store.setModelType('decision_tree_classifier');
          store.setParam('treeDepth', 4);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.25);
          store.regenerateDataset();
        },
      },
      {
        id: 'cls-forest-stability',
        title: 'Forest Stability',
        prompt: 'Use random forests to stabilize single-tree fluctuations.',
        expected: 'Validation metrics should fluctuate less than a single tree.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_overlap');
          store.setModelType('random_forest_classifier');
          store.setParam('treeDepth', 5);
          store.setParam('forestTrees', 45);
          store.setEvaluationMode('cross_validation');
          store.setCvFolds(5);
          store.regenerateDataset();
        },
      },
    ],
    [store]
  );

  const plannedMissions: PlannedMission[] = [
    {
      id: 'threshold-tradeoff',
      title: 'Boosting Focus',
      prompt: 'Track how boosting reweights hard examples across iterations.',
      expected: 'Boundary focus shifts to hard regions.',
      phase: 'Phase 3',
    },
    {
      id: 'false-positive-control',
      title: 'Feature Importance Lens',
      prompt: 'Compare global vs local importance behavior across ensemble models.',
      expected: 'Interpretability tradeoffs become explicit.',
      phase: 'Phase 3',
    },
  ];

  const applyMission = (mission: ActiveMission) => {
    mission.apply();
    setPulseMission(mission.id);
    window.setTimeout(() => setPulseMission(null), 1400);
  };

  const activeMissions = taskMode === 'classification' ? classificationMissions : regressionMissions;

  const recommendedMissions = activeMissions.slice(0, 2);
  const visibleMissions = compact && !showAllCompact ? recommendedMissions : activeMissions;

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <p className="panel-title">Learning Missions</p>
        {compact ? (
          <button type="button" onClick={() => setShowAllCompact((value) => !value)} className="quick-action">
            <span>{showAllCompact ? 'Show Less' : 'See All'}</span>
          </button>
        ) : (
          <span className="text-[11px] text-text-tertiary">Self-paced, no scoring</span>
        )}
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {visibleMissions.map((mission) => (
          <button
            key={mission.id}
            type="button"
            onClick={() => applyMission(mission)}
            className={`mission-card text-left ${pulseMission === mission.id ? 'mission-card-success' : ''}`}
          >
            <div className="flex items-center justify-between gap-2 mb-1">
              <p className="text-sm font-medium text-text-primary">{mission.title}</p>
              <span className="inline-flex items-center gap-1 text-[11px] text-emerald-500 dark:text-emerald-300">
                <Sparkles className="w-3 h-3" />
                Now Available
              </span>
            </div>
            <p className="text-xs text-text-secondary leading-snug">{mission.prompt}</p>
            <p className="text-[11px] text-text-tertiary mt-1.5">{mission.expected}</p>
          </button>
        ))}
      </div>

      {!compact && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {plannedMissions.map((mission) => (
            <div key={mission.id} className="mission-card mission-card-locked">
              <div className="flex items-center justify-between gap-2 mb-1">
                <p className="text-sm font-medium text-text-primary">{mission.title}</p>
                <span className="inline-flex items-center gap-1 text-[11px] text-text-tertiary">
                  <Lock className="w-3 h-3" />
                  {mission.phase}
                </span>
              </div>
              <p className="text-xs text-text-secondary leading-snug">{mission.prompt}</p>
              <p className="text-[11px] text-text-tertiary mt-1.5">{mission.expected}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
