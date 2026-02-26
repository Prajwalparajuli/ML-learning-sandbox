import { useMemo, useState } from 'react';
import { Sparkles } from 'lucide-react';
import { useModelStore } from '../store/modelStore';

type ActiveMission = {
  id: string;
  title: string;
  prompt: string;
  expected: string;
  tier: 'starter' | 'analyst' | 'expert';
  criteria: string;
  isComplete: () => boolean;
  rationale: string;
  apply: () => void;
};

interface ScenarioMissionsProps {
  compact?: boolean;
}

export function ScenarioMissions({ compact = false }: ScenarioMissionsProps) {
  const [pulseMission, setPulseMission] = useState<string | null>(null);
  const [showAllCompact, setShowAllCompact] = useState(false);
  const store = useModelStore();
  const taskMode = store.taskMode;
  const missionTier = store.missionTier;
  const setMissionTier = store.setMissionTier;

  const regressionMissions: ActiveMission[] = useMemo(
    () => [
      {
        id: 'fix-overfit',
        title: 'Fix Overfit',
        prompt: 'Start from a too-flexible setup and stabilize validation behavior.',
        expected: 'Validation error curve should flatten and metric volatility should drop.',
        tier: 'expert',
        criteria: 'RMSE < 2.20 and compare-vs-OLS enabled',
        isComplete: () => store.metrics.rmse < 2.2 && store.compareWithOls,
        rationale: 'Reduced complexity should improve generalization stability.',
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
        tier: 'starter',
        criteria: 'Model is polynomial degree >= 3 on quadratic dataset',
        isComplete: () => store.modelType === 'polynomial' && store.params.polynomialDegree >= 3 && store.dataset === 'quadratic',
        rationale: 'Higher-order basis reduces bias on curved data.',
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
        tier: 'analyst',
        criteria: 'Diagnostics enabled and compare-vs-OLS enabled',
        isComplete: () => store.showAssumptions && store.compareWithOls,
        rationale: 'Diagnostics should guide robust model choice under outliers.',
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
        tier: 'analyst',
        criteria: 'Ridge alpha >= 0.8 and cross-validation mode',
        isComplete: () => store.modelType === 'ridge' && store.params.alpha >= 0.8 && store.evaluationMode === 'cross_validation',
        rationale: 'Controlled shrinkage usually trades variance for stability.',
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
        tier: 'starter',
        criteria: 'SVM on overlap dataset in train/test mode',
        isComplete: () => store.modelType === 'svm_classifier' && store.dataset === 'class_overlap' && store.evaluationMode === 'train_test',
        rationale: 'Margin tuning should alter boundary geometry and errors.',
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
        tier: 'analyst',
        criteria: 'Recall >= 0.70 on imbalanced data',
        isComplete: () => store.dataset === 'class_imbalanced' && store.metrics.recall >= 0.7,
        rationale: 'Threshold control exposes precision-recall tradeoffs.',
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
        tier: 'starter',
        criteria: 'Decision threshold >= 0.65 in logistic model',
        isComplete: () => store.modelType === 'logistic_classifier' && store.params.decisionThreshold >= 0.65,
        rationale: 'Threshold controls classification operating point.',
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
        tier: 'analyst',
        criteria: 'KNN with k between 5 and 9 on moons',
        isComplete: () => store.modelType === 'knn_classifier' && store.dataset === 'class_moons' && store.params.knnK >= 5 && store.params.knnK <= 9,
        rationale: 'Neighborhood size governs local boundary flexibility.',
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
        tier: 'expert',
        criteria: 'Tree depth >= 4 and FP + FN <= 16',
        isComplete: () => store.modelType === 'decision_tree_classifier' && store.params.treeDepth >= 4 && (1 - store.metrics.accuracy) * store.data.length <= 16,
        rationale: 'Tree depth can improve fit but may amplify instability.',
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
        tier: 'expert',
        criteria: 'Forest trees >= 40 and F1 >= 0.75',
        isComplete: () => store.modelType === 'random_forest_classifier' && store.params.forestTrees >= 40 && store.metrics.f1 >= 0.75,
        rationale: 'Bagging reduces variance versus single-tree partitions.',
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
      {
        id: 'cls-boosting-focus',
        title: 'Boosting Focus',
        prompt: 'Track how boosting emphasizes harder examples across rounds.',
        expected: 'Local errors should reduce while boundary focus shifts.',
        tier: 'expert',
        criteria: 'AdaBoost or Gradient Boosting with rounds >= 40',
        isComplete: () => (store.modelType === 'adaboost_classifier' || store.modelType === 'gradient_boosting_classifier') && store.params.boostingRounds >= 40,
        rationale: 'Boosting reallocates capacity toward difficult regions.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_overlap');
          store.setModelType('gradient_boosting_classifier');
          store.setParam('boostingRounds', 50);
          store.setParam('learningRate', 0.15);
          store.setEvaluationMode('cross_validation');
          store.setCvFolds(5);
          store.regenerateDataset();
        },
      },
      {
        id: 'cls-feature-importance-lens',
        title: 'Feature Importance Lens',
        prompt: 'Compare tree vs boosting behavior on the same overlap data.',
        expected: 'Global split behavior changes with ensemble strategy.',
        tier: 'expert',
        criteria: 'Random Forest or Gradient Boosting with diagnostics on',
        isComplete: () => (store.modelType === 'random_forest_classifier' || store.modelType === 'gradient_boosting_classifier') && store.showClassificationDiagnostics,
        rationale: 'Ensemble type changes how feature effects aggregate.',
        apply: () => {
          store.setTaskMode('classification');
          store.setDataset('class_overlap');
          store.setModelType('random_forest_classifier');
          store.setParam('treeDepth', 5);
          store.setParam('forestTrees', 60);
          store.setShowClassificationDiagnostics(true);
          store.setEvaluationMode('train_test');
          store.setTestRatio(0.25);
          store.regenerateDataset();
        },
      },
    ],
    [store]
  );

  const applyMission = (mission: ActiveMission) => {
    mission.apply();
    setPulseMission(mission.id);
    window.setTimeout(() => setPulseMission(null), 1400);
  };

  const activeMissions = taskMode === 'classification' ? classificationMissions : regressionMissions;

  const tierFiltered = activeMissions.filter((mission) => mission.tier === missionTier);
  const recommendedMissions = tierFiltered.slice(0, 2);
  const visibleMissions = compact && !showAllCompact ? recommendedMissions : tierFiltered;

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <p className="panel-title">Learning Missions</p>
        <div className="hidden sm:flex gap-1">
          {(['starter', 'analyst', 'expert'] as const).map((tier) => (
            <button key={tier} type="button" className={`quick-action ${missionTier === tier ? 'quick-action-active' : ''}`} onClick={() => setMissionTier(tier)}>
              {tier}
            </button>
          ))}
        </div>
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
              <Sparkles className="w-3.5 h-3.5 text-amber-500/80" />
            </div>
            <p className="text-xs text-text-secondary leading-snug">{mission.prompt}</p>
            <p className="text-[11px] text-text-tertiary mt-1.5">{mission.expected}</p>
            <p className="text-[11px] text-text-tertiary mt-1">{mission.criteria}</p>
            <p className={`text-[11px] mt-1 ${mission.isComplete() ? 'text-emerald-400' : 'text-text-tertiary'}`}>
              {mission.isComplete() ? `Completed: ${mission.rationale}` : 'In progress'}
            </p>
          </button>
        ))}
      </div>

    </div>
  );
}
