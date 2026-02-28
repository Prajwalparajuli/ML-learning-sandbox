import { useModelStore, type ModelParams, type ModelType } from '../store/modelStore';
import { TrendingUp, BarChart2, Activity, Target } from 'lucide-react';
import { evaluateModelMetrics } from '../lib/dataUtils';
import { useEffect, useMemo, useRef, useState } from 'react';
import { metricMeta } from '../content/classicalContentAdapter';
import { featureFlags } from '../config/featureFlags';
import { InfoPopover } from './InfoPopover';

interface MetricsBoardProps {
  compact?: boolean;
}

export function MetricsBoard({ compact = false }: MetricsBoardProps) {
  const {
    taskMode,
    metrics,
    data,
    compareWithOls,
    comparePinnedModels,
    clearPinnedModels,
    params,
    evaluationMode,
    testRatio,
    cvFolds,
    randomSeed,
    datasetVersion,
    selectedMetrics,
  } = useModelStore();
  const prevMetricsRef = useRef(metrics);
  const [deltaCaption, setDeltaCaption] = useState<string>('Adjust one control to see what changed.');

  const paramsForModel = (target: ModelType, base: ModelParams): ModelParams => {
    const defaults: ModelParams = {
      alpha: 0.1,
      l1Ratio: 0.5,
      polynomialDegree: 3,
      stepwiseTerms: 2,
      knnK: 5,
      svmC: 1,
      svmGamma: 1,
      svmEpsilon: 0.1,
      treeDepth: 4,
      forestTrees: 35,
      boostingRounds: 40,
      learningRate: 0.1,
      pcaComponents: 2,
      plsComponents: 2,
      decisionThreshold: 0.5,
    };
    if (target === 'ridge' || target === 'lasso' || target === 'elasticnet' || target === 'svm_regressor' || target === 'svm_classifier') {
      defaults.alpha = base.alpha;
      defaults.svmC = base.svmC;
      defaults.svmGamma = base.svmGamma;
      defaults.svmEpsilon = base.svmEpsilon;
    }
    if (target === 'polynomial' || target === 'forward_stepwise' || target === 'backward_stepwise') {
      defaults.polynomialDegree = base.polynomialDegree;
      defaults.stepwiseTerms = base.stepwiseTerms;
    }
    if (target === 'knn_classifier') defaults.knnK = base.knnK;
    if (target === 'decision_tree_classifier' || target === 'random_forest_classifier') defaults.treeDepth = base.treeDepth;
    if (target === 'random_forest_classifier') defaults.forestTrees = base.forestTrees;
    if (target === 'adaboost_classifier' || target === 'gradient_boosting_classifier') {
      defaults.boostingRounds = base.boostingRounds;
      defaults.learningRate = base.learningRate;
    }
    if (target === 'pcr_regressor') defaults.pcaComponents = base.pcaComponents;
    if (target === 'pls_regressor') defaults.plsComponents = base.plsComponents;
    defaults.decisionThreshold = base.decisionThreshold;
    return defaults;
  };

  const olsMetrics = useMemo(
    () => (compareWithOls && taskMode === 'regression' && data.length > 0
      ? evaluateModelMetrics(data, 'ols', params, evaluationMode, testRatio, cvFolds, randomSeed + datasetVersion)
      : null),
    [compareWithOls, taskMode, data, params, evaluationMode, testRatio, cvFolds, randomSeed, datasetVersion]
  );

  const activeMetrics = metrics;
  const pinnedComparisons = useMemo(
    () => comparePinnedModels.map((pinned) => ({
      model: pinned,
      metrics: evaluateModelMetrics(
        data,
        pinned,
        paramsForModel(pinned, params),
        evaluationMode,
        testRatio,
        cvFolds,
        randomSeed + datasetVersion
      ),
    })),
    [comparePinnedModels, data, params, evaluationMode, testRatio, cvFolds, randomSeed, datasetVersion]
  );

  useEffect(() => {
    const prev = prevMetricsRef.current;
    if (!prev) {
      prevMetricsRef.current = metrics;
      return;
    }
    let nextCaption = 'Adjust one control to see what changed.';
    if (taskMode === 'regression') {
      const r2Delta = metrics.r2 - prev.r2;
      const rmseDelta = metrics.rmse - prev.rmse;
      if (r2Delta > 0.01 && rmseDelta < -0.01) {
        nextCaption = 'Fit improved: explained variance rose while error fell.';
      } else if (r2Delta < -0.01 && rmseDelta > 0.01) {
        nextCaption = 'Generalization weakened: variance explained dropped and error increased.';
      } else if (Math.abs(r2Delta) < 0.005 && Math.abs(rmseDelta) < 0.005) {
        nextCaption = 'Change impact is small: model behavior is currently stable.';
      } else if (r2Delta > 0 && rmseDelta > 0) {
        nextCaption = 'Tradeoff detected: fit score improved but absolute errors rose on this split.';
      } else {
        nextCaption = 'Bias-variance balance shifted; check diagnostics before locking this setting.';
      }
    } else {
      const recallDelta = metrics.recall - prev.recall;
      const precisionDelta = metrics.precision - prev.precision;
      if (recallDelta > 0.01 && precisionDelta < -0.01) {
        nextCaption = 'Boundary became more permissive: recall rose while precision fell.';
      } else if (recallDelta < -0.01 && precisionDelta > 0.01) {
        nextCaption = 'Boundary became more selective: precision improved while recall dropped.';
      } else if (metrics.f1 - prev.f1 > 0.01) {
        nextCaption = 'Class separation improved: F1 increased with the latest change.';
      } else if (Math.abs(metrics.f1 - prev.f1) < 0.005) {
        nextCaption = 'Little movement in class quality metrics from the last adjustment.';
      } else {
        nextCaption = 'Threshold-quality tradeoff shifted; inspect confidence diagnostics next.';
      }
    }
    const timer = window.setTimeout(() => setDeltaCaption(nextCaption), 0);
    prevMetricsRef.current = metrics;
    return () => window.clearTimeout(timer);
  }, [metrics, taskMode]);

  const regressionCards = [
    {
      key: 'r2' as const,
      label: 'R² Score',
      value: activeMetrics.r2,
      icon: TrendingUp,
      description: metricMeta.r2.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'adjustedR2' as const,
      label: 'Adjusted R²',
      value: activeMetrics.adjustedR2,
      icon: TrendingUp,
      description: metricMeta.adjustedR2.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'rmse' as const,
      label: 'RMSE',
      value: activeMetrics.rmse,
      icon: BarChart2,
      description: metricMeta.rmse.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'mae' as const,
      label: 'MAE',
      value: activeMetrics.mae,
      icon: Activity,
      description: metricMeta.mae.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'mse' as const,
      label: 'MSE',
      value: activeMetrics.mse,
      icon: Target,
      description: metricMeta.mse.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'mape' as const,
      label: 'MAPE (%)',
      value: activeMetrics.mape,
      icon: Activity,
      description: metricMeta.mape.description,
      format: (v: number) => v.toFixed(2),
    },
    {
      key: 'medianAe' as const,
      label: 'Median AE',
      value: activeMetrics.medianAe,
      icon: Target,
      description: metricMeta.medianAe.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'explainedVariance' as const,
      label: 'Expl. Variance',
      value: activeMetrics.explainedVariance,
      icon: TrendingUp,
      description: metricMeta.explainedVariance.description,
      format: (v: number) => v.toFixed(4),
    },
  ];
  const classificationCards = [
    {
      key: 'accuracy' as const,
      label: 'Accuracy',
      value: activeMetrics.accuracy,
      icon: TrendingUp,
      description: metricMeta.accuracy.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'precision' as const,
      label: 'Precision',
      value: activeMetrics.precision,
      icon: Target,
      description: metricMeta.precision.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'recall' as const,
      label: 'Recall',
      value: activeMetrics.recall,
      icon: Activity,
      description: metricMeta.recall.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'f1' as const,
      label: 'F1 Score',
      value: activeMetrics.f1,
      icon: TrendingUp,
      description: metricMeta.f1.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'specificity' as const,
      label: 'Specificity',
      value: activeMetrics.specificity,
      icon: Target,
      description: metricMeta.specificity.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'rocAuc' as const,
      label: 'ROC-AUC',
      value: activeMetrics.rocAuc,
      icon: BarChart2,
      description: metricMeta.rocAuc.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'prAuc' as const,
      label: 'PR-AUC',
      value: activeMetrics.prAuc,
      icon: BarChart2,
      description: metricMeta.prAuc.description,
      format: (v: number) => v.toFixed(4),
    },
    {
      key: 'logLoss' as const,
      label: 'Log Loss',
      value: activeMetrics.logLoss,
      icon: Activity,
      description: metricMeta.logLoss.description,
      format: (v: number) => v.toFixed(4),
    },
  ];
  const metricCards = taskMode === 'classification' ? classificationCards : regressionCards;
  const visibleCards = metricCards.filter((card) => selectedMetrics.includes(card.key));

  return (
    <div className="space-y-3">
      <div className="metric-mode-indicator">
        {evaluationMode === 'full' && 'Evaluation: Full dataset fit'}
        {evaluationMode === 'train_test' && `Evaluation: Train/Test (${Math.round((1 - testRatio) * 100)}% / ${Math.round(testRatio * 100)}%)`}
        {evaluationMode === 'cross_validation' && `Evaluation: ${cvFolds}-Fold Cross-Validation`}
      </div>
      {taskMode === 'regression' && compareWithOls && (
        <div className="material-panel-soft px-3 py-2 text-xs text-text-secondary">
          Comparing active configuration against OLS baseline on the same dataset.
        </div>
      )}
      <div className="grid grid-cols-2 gap-2.5">
        {visibleCards.map((card) => {
          const baselineValue = taskMode === 'regression' && olsMetrics ? olsMetrics[card.key] : null;
          const delta = baselineValue !== null ? card.value - baselineValue : null;
          const higherIsBetter = card.key === 'r2' || card.key === 'adjustedR2' || card.key === 'explainedVariance';

          return (
            <div
              key={card.label}
              className="metric-card p-2.5"
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <card.icon className="w-4 h-4 text-text-tertiary" />
                <span className="text-xs text-text-secondary">{card.label}</span>
                {featureFlags.ff_info_microcards && (
                  <InfoPopover
                    label={`${card.label} Insight`}
                    what={card.description}
                    why="Metric trends indicate how each model balances fit quality and generalization."
                    tryNext="Change one control at a time and compare this metric against OLS baseline."
                  />
                )}
              </div>
              <div className="text-lg font-mono font-semibold text-accent">
                {card.format(card.value)}
              </div>
              {!compact && <div className="text-xs text-text-tertiary mt-1">{card.description}</div>}
              {delta !== null && (
                <div className={`mt-2 text-[11px] font-medium ${
                  higherIsBetter
                    ? delta >= 0 ? 'text-emerald-400' : 'text-rose-400'
                    : delta <= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}>
                  {delta >= 0 ? '+' : ''}{delta.toFixed(4)} vs OLS
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="material-panel-soft px-3 py-2 text-xs text-text-secondary">
        Why this changed: {deltaCaption}
      </div>
      {comparePinnedModels.length > 0 && (
        <div className="material-panel-soft p-2.5">
          <div className="flex items-center justify-between mb-2">
            <p className="panel-title">Model Compare Tray</p>
            <button type="button" className="text-[11px] text-text-tertiary hover:text-text-primary" onClick={clearPinnedModels}>
              Clear
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {pinnedComparisons.map((entry) => (
              <div key={entry.model} className="rounded-xl border border-border-subtle p-2">
                <div className="text-xs font-semibold text-text-primary mb-1.5">{entry.model.replaceAll('_', ' ')}</div>
                <div className="text-[11px] text-text-secondary">
                  {taskMode === 'classification'
                    ? `Accuracy ${entry.metrics.accuracy.toFixed(3)} · F1 ${entry.metrics.f1.toFixed(3)} · ROC-AUC ${entry.metrics.rocAuc.toFixed(3)}`
                    : `R² ${entry.metrics.r2.toFixed(3)} · RMSE ${entry.metrics.rmse.toFixed(3)} · MAE ${entry.metrics.mae.toFixed(3)}`}
                </div>
              </div>
            ))}
            {comparePinnedModels.length < 2 && (
              <div className="rounded-xl border border-dashed border-border-subtle p-2 text-[11px] text-text-tertiary">
                Pin up to 2 models from the hero strip to compare on the same data and seed.
              </div>
            )}
          </div>
        </div>
      )}
      {taskMode === 'regression' && compareWithOls && (
        <div className="rounded-xl border border-dashed border-border-subtle px-3 py-2 text-[11px] text-text-tertiary">
          OLS baseline uses closed-form fit from current dataset ({data.length} points).
        </div>
      )}
    </div>
  );
}
