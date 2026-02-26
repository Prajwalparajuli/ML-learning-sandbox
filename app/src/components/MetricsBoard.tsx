import { useModelStore } from '../store/modelStore';
import { TrendingUp, BarChart2, Activity, Target } from 'lucide-react';
import { evaluateModelMetrics } from '../lib/dataUtils';
import { useMemo } from 'react';
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
    params,
    evaluationMode,
    testRatio,
    cvFolds,
    randomSeed,
    datasetVersion,
    selectedMetrics,
  } = useModelStore();

  const olsMetrics = useMemo(
    () => (compareWithOls && taskMode === 'regression' && data.length > 0
      ? evaluateModelMetrics(data, 'ols', params, evaluationMode, testRatio, cvFolds, randomSeed + datasetVersion)
      : null),
    [compareWithOls, taskMode, data, params, evaluationMode, testRatio, cvFolds, randomSeed, datasetVersion]
  );

  const activeMetrics = metrics;

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
      {taskMode === 'regression' && compareWithOls && (
        <div className="rounded-xl border border-dashed border-border-subtle px-3 py-2 text-[11px] text-text-tertiary">
          OLS baseline uses closed-form fit from current dataset ({data.length} points).
        </div>
      )}
    </div>
  );
}
