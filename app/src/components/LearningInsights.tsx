import Plot from 'react-plotly.js';
import { useMemo } from 'react';
import { useTheme } from 'next-themes';
import { useModelStore } from '../store/modelStore';
import { computeBiasVarianceCurve, computeClassificationComplexityCurve } from '../lib/dataUtils';
import { biasVarianceNarrative } from '../content/classicalContentAdapter';

interface LearningInsightsProps {
  sidebarCollapsed?: boolean;
  compact?: boolean;
}

export function LearningInsights({ sidebarCollapsed = false, compact = false }: LearningInsightsProps) {
  const { taskMode, data, modelType, params, randomSeed, datasetVersion, featureMode } = useModelStore();
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  const axisText = isDark ? '#94a3b8' : '#475569';
  const grid = isDark ? 'rgba(100,116,139,0.22)' : 'rgba(148,163,184,0.35)';
  const legendBg = isDark ? 'rgba(30,41,59,0.78)' : 'rgba(255,255,255,0.84)';
  const legendBorder = isDark ? 'rgba(100,116,139,0.45)' : 'rgba(148,163,184,0.5)';

  const curve = useMemo(
    () => computeBiasVarianceCurve(data, randomSeed + datasetVersion, featureMode),
    [data, randomSeed, datasetVersion, featureMode]
  );
  const clsCurve = useMemo(
    () => computeClassificationComplexityCurve(data, modelType, params, randomSeed + datasetVersion),
    [data, modelType, params, randomSeed, datasetVersion]
  );

  if (taskMode === 'classification') {
    return (
      <div className={compact ? 'space-y-2.5' : 'space-y-3'}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <div className="story-card">
            <p className="panel-title mb-0.5">Boundary Flexibility</p>
            <p className="text-xs leading-snug text-text-secondary">Higher complexity often lowers train loss first, then risks validation drift.</p>
          </div>
          <div className="story-card">
            <p className="panel-title mb-0.5">Threshold Tradeoff</p>
            <p className="text-xs leading-snug text-text-secondary">Threshold shifts confusion outcomes; complexity shifts boundary capacity.</p>
          </div>
          <div className="story-card">
            <p className="panel-title mb-0.5">Generalization Check</p>
            <p className="text-xs leading-snug text-text-secondary">Prefer complexity settings where validation loss is low and stable.</p>
          </div>
        </div>
        {clsCurve.complexity.length > 0 && (
          <div className="plot-wrap code-block h-64 overflow-hidden">
            <Plot
              key={`cls-complexity-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
              data={[
                {
                  x: clsCurve.complexity,
                  y: clsCurve.trainLogLoss,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Train LogLoss',
                  line: { color: '#2563eb', width: 2.2 },
                  marker: { size: 6 },
                },
                {
                  x: clsCurve.complexity,
                  y: clsCurve.validationLogLoss,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Validation LogLoss',
                  line: { color: '#f59e0b', width: 2.2 },
                  marker: { size: 6 },
                },
                {
                  x: clsCurve.complexity,
                  y: clsCurve.validationF1,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Validation F1',
                  yaxis: 'y2',
                  line: { color: '#10b981', width: 1.8, dash: 'dot' },
                  marker: { size: 5 },
                },
              ]}
              layout={{
                margin: { l: 46, r: 44, t: 16, b: 42 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: axisText, family: 'Inter, sans-serif' },
                xaxis: { title: { text: clsCurve.label }, color: axisText, gridcolor: grid },
                yaxis: { title: { text: 'Log Loss' }, color: axisText, gridcolor: grid },
                yaxis2: { title: { text: 'F1' }, overlaying: 'y', side: 'right', color: axisText, range: [0, 1] },
                legend: { x: 0.02, y: 0.98, bgcolor: legendBg, bordercolor: legendBorder, borderwidth: 1, font: { size: 11, color: axisText } },
              }}
              config={{ responsive: true, displayModeBar: false }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        )}
        {clsCurve.complexity.length === 0 && (
          <div className="story-card">
            <p className="text-xs leading-snug text-text-secondary">Increase sample size or switch to train/test or cross-validation to unlock complexity lens.</p>
          </div>
        )}
      </div>
    );
  }

  if (curve.complexity.length === 0) return null;

  const minValidationMse = Math.min(...curve.validationMse);
  const bestIndex = curve.validationMse.findIndex((value) => value === minValidationMse);
  const bestComplexity = curve.complexity[bestIndex];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        <div className="story-card">
          <p className="panel-title mb-0.5">Underfitting</p>
          <p className="text-xs leading-snug text-text-secondary">{biasVarianceNarrative.underfitting}</p>
        </div>
        <div className="story-card">
          <p className="panel-title mb-0.5">Overfitting</p>
          <p className="text-xs leading-snug text-text-secondary">{biasVarianceNarrative.overfitting}</p>
        </div>
        <div className="story-card">
          <p className="panel-title mb-0.5">Best Complexity</p>
          <p className="text-xs leading-snug text-text-secondary">
            Validation MSE is lowest at complexity level {bestComplexity}. {biasVarianceNarrative.sweetSpot}
          </p>
        </div>
      </div>

      <div className="plot-wrap code-block h-64 overflow-hidden">
        <Plot
          key={`bias-variance-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
          data={[
            {
              x: curve.complexity,
              y: curve.trainMse,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Train MSE',
              line: { color: '#2563eb', width: 2.5 },
              marker: { size: 7 },
            },
            {
              x: curve.complexity,
              y: curve.validationMse,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Validation MSE',
              line: { color: '#f59e0b', width: 2.5 },
              marker: { size: 7 },
            },
          ]}
          layout={{
            margin: { l: 46, r: 14, t: 16, b: 42 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: axisText, family: 'Inter, sans-serif' },
            xaxis: {
              title: { text: featureMode === '1d' ? 'Model Complexity (Degree)' : 'Complexity (Lower alpha → More flexible)' },
              color: axisText,
              gridcolor: grid,
            },
            yaxis: {
              title: { text: 'Mean Squared Error' },
              color: axisText,
              gridcolor: grid,
            },
            legend: {
              x: 0.02,
              y: 0.98,
              bgcolor: legendBg,
              bordercolor: legendBorder,
              borderwidth: 1,
              font: { size: 11, color: axisText },
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
}
