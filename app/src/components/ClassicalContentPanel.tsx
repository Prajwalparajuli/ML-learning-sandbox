import { useMemo, useState } from 'react';
import { PlayCircle, Gauge, FlaskConical, Shuffle, ChevronDown, ChevronUp } from 'lucide-react';
import { useModelStore } from '../store/modelStore';
import { diagnosticsMeta, metricMeta, modelContentMap, resamplingMeta } from '../content/classicalContentAdapter';
import { toast } from 'sonner';

interface ClassicalContentPanelProps {
  compact?: boolean;
}

export function ClassicalContentPanel({ compact = false }: ClassicalContentPanelProps) {
  const [showMore, setShowMore] = useState(false);
  const {
    taskMode,
    modelType,
    selectedMetrics,
    evaluationMode,
    dataset,
    setTaskMode,
    setModelType,
    setDataset,
    setEvaluationMode,
    setCvFolds,
    setTestRatio,
    setParam,
    regenerateDataset,
    captureSandboxSnapshot,
    restoreSandboxSnapshot,
  } = useModelStore();
  const activeModel = modelContentMap[modelType];
  const activeResampling = resamplingMeta.find((item) => item.implementedModes.includes(evaluationMode));
  const primaryMetric = selectedMetrics[0];
  const tryNextActions = taskMode === 'classification'
    ? [
      {
        id: 'cls-threshold-shift',
        title: 'Threshold Sprint',
        subtitle: 'Lower threshold to prioritize recall',
        run: () => {
          setTaskMode('classification');
          setModelType('logistic_classifier');
          setDataset('class_imbalanced');
          setEvaluationMode('train_test');
          setTestRatio(0.3);
          setParam('decisionThreshold', 0.35);
          regenerateDataset();
        },
      },
      {
        id: 'cls-tree-compare',
        title: 'Tree vs Forest',
        subtitle: 'Compare single-tree variance vs bagging',
        run: () => {
          setTaskMode('classification');
          setDataset('class_overlap');
          setModelType('random_forest_classifier');
          setParam('treeDepth', 5);
          setParam('forestTrees', 45);
          setEvaluationMode('cross_validation');
          setCvFolds(5);
          regenerateDataset();
        },
      },
      {
        id: 'cls-margin',
        title: 'Margin Sweep',
        subtitle: 'Raise C to tighten the SVM boundary',
        run: () => {
          setTaskMode('classification');
          setDataset('class_moons');
          setModelType('svm_classifier');
          setParam('svmC', 2.5);
          setParam('svmGamma', 0.8);
          setParam('decisionThreshold', 0.5);
          regenerateDataset();
        },
      },
      {
        id: 'cls-boost',
        title: 'Boost Focus',
        subtitle: 'Try boosted stumps on overlap',
        run: () => {
          setTaskMode('classification');
          setDataset('class_overlap');
          setModelType('adaboost_classifier');
          setParam('boostingRounds', 70);
          setParam('learningRate', 0.2);
          setEvaluationMode('cross_validation');
          setCvFolds(5);
          regenerateDataset();
        },
      },
    ]
    : [
      {
        id: 'reg-quick-ab',
        title: 'Quick A/B',
        subtitle: 'Switch OLS to Ridge on noisy data',
        run: () => {
          setTaskMode('regression');
          setDataset('noisy');
          setModelType('ridge');
          setParam('alpha', 0.35);
          setEvaluationMode('train_test');
          setTestRatio(0.25);
          regenerateDataset();
        },
      },
      {
        id: 'reg-sensitivity',
        title: 'Sensitivity',
        subtitle: 'Sweep alpha for stability',
        run: () => {
          setTaskMode('regression');
          setDataset('heteroscedastic');
          setModelType('ridge');
          setParam('alpha', 0.85);
          setEvaluationMode('cross_validation');
          setCvFolds(6);
          regenerateDataset();
        },
      },
      {
        id: 'reg-diagnostics',
        title: 'Diagnostics',
        subtitle: 'Stress assumptions with outliers',
        run: () => {
          setTaskMode('regression');
          setDataset('outliers');
          setModelType('ols');
          setEvaluationMode('train_test');
          setTestRatio(0.3);
          regenerateDataset();
        },
      },
      {
        id: 'reg-resample',
        title: 'Resampling',
        subtitle: activeResampling ? `Current mode: ${activeResampling.title}` : 'Compare train/test vs k-fold stability',
        run: () => {
          setTaskMode('regression');
          setDataset('noisy');
          setModelType('elasticnet');
          setParam('alpha', 0.45);
          setParam('l1Ratio', 0.35);
          setEvaluationMode('cross_validation');
          setCvFolds(5);
          regenerateDataset();
        },
      },
    ];

  const insights = useMemo(() => {
    const metricNote = primaryMetric ? metricMeta[primaryMetric]?.description : 'Track metrics while changing one control at a time.';
    return [
      `Model cue: ${activeModel.explanation}`,
      `Dataset cue: ${dataset} responds differently as complexity changes.`,
      `Evaluation cue: ${activeResampling?.title ?? evaluationMode} keeps generalization checks grounded.`,
      `Metric cue: ${metricNote}`,
    ];
  }, [activeModel.explanation, dataset, activeResampling?.title, evaluationMode, primaryMetric]);

  const applyTryNext = (action: (typeof tryNextActions)[number]) => {
    const snapshot = captureSandboxSnapshot();
    action.run();
    toast.success('Preset applied', {
      action: {
        label: 'Revert',
        onClick: () => restoreSandboxSnapshot(snapshot),
      },
    });
  };

  return (
    <div className="space-y-2.5">
      <div className="material-panel-soft p-2.5">
        <p className="text-xs text-text-secondary">
          Learning mode: <span className="text-text-primary font-medium">{activeModel.title}</span> on{' '}
          <span className="text-text-primary font-medium">{dataset}</span>.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        {insights.slice(0, 3).map((insight) => (
          <div key={insight} className="story-card p-2.5">
            <p className="text-xs text-text-secondary leading-snug">{insight}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {tryNextActions.map((action, index) => (
          <button
            key={action.id}
            type="button"
            onClick={() => applyTryNext(action)}
            className="story-card p-2.5 text-left interactive-lift focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
          >
            <p className="panel-title mb-1 flex items-center gap-1">
              {index % 4 === 0 && <PlayCircle className="w-3.5 h-3.5" />}
              {index % 4 === 1 && <Gauge className="w-3.5 h-3.5" />}
              {index % 4 === 2 && <FlaskConical className="w-3.5 h-3.5" />}
              {index % 4 === 3 && <Shuffle className="w-3.5 h-3.5" />}
              {action.title}
            </p>
            <p className="text-xs text-text-secondary">{action.subtitle}</p>
          </button>
        ))}
      </div>

      {!compact && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setShowMore((value) => !value)}
            className="quick-action"
          >
            {showMore ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            <span>{showMore ? 'Hide More Theory' : 'More Theory'}</span>
          </button>
          {showMore && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {diagnosticsMeta.slice(0, 4).map((diagnostic) => (
                <div key={diagnostic.id} className="story-card p-2.5">
                  <p className="text-sm font-medium text-text-primary">{diagnostic.title}</p>
                  <p className="text-xs text-text-secondary mt-1">{diagnostic.detail}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
