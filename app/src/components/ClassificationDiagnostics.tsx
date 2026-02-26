import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { useTheme } from 'next-themes';
import { useModelStore } from '../store/modelStore';
import { computeClassificationDiagnostics, fitRegressionModel, splitDataset } from '../lib/dataUtils';
import { InfoPopover } from './InfoPopover';
import { featureFlags } from '../config/featureFlags';

interface ClassificationDiagnosticsProps {
  sidebarCollapsed?: boolean;
  compact?: boolean;
}

export function ClassificationDiagnostics({ sidebarCollapsed = false, compact = false }: ClassificationDiagnosticsProps) {
  const [showConfidenceView, setShowConfidenceView] = useState(false);
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  const axisText = isDark ? '#94a3b8' : '#475569';
  const grid = isDark ? 'rgba(100,116,139,0.24)' : 'rgba(148,163,184,0.35)';
  const zero = isDark ? 'rgba(148,163,184,0.35)' : 'rgba(100,116,139,0.35)';
  const legendBg = isDark ? 'rgba(30,41,59,0.78)' : 'rgba(255,255,255,0.82)';
  const legendBorder = isDark ? 'rgba(100,116,139,0.45)' : 'rgba(148,163,184,0.5)';

  const { data, modelType, params, evaluationMode, testRatio, randomSeed, datasetVersion } = useModelStore();
  const threshold = Math.min(0.95, Math.max(0.05, params.decisionThreshold));
  const split = useMemo(
    () => (evaluationMode === 'train_test'
      ? splitDataset(data, testRatio, randomSeed + datasetVersion, true)
      : { train: data, test: [] as typeof data }),
    [data, evaluationMode, testRatio, randomSeed, datasetVersion]
  );

  const diagnostic = useMemo(() => {
    if (data.length === 0) return null;
    const evalSet = evaluationMode === 'train_test' ? split.test : data;
    const trainSet = evaluationMode === 'train_test' ? split.train : data;
    if (evalSet.length === 0 || trainSet.length === 0) return null;
    const fit = fitRegressionModel(trainSet, modelType, params);
    const yTrue = evalSet.map((point) => point.y);
    const yProb = evalSet.map((point) => fit.predict(point.features));
    return computeClassificationDiagnostics(yTrue, yProb, threshold);
  }, [data, evaluationMode, split, modelType, params, threshold]);

  const confidenceDiagnostics = useMemo(() => {
    const evalSet = evaluationMode === 'train_test' ? split.test : data;
    const trainSet = evaluationMode === 'train_test' ? split.train : data;
    if (evalSet.length === 0 || trainSet.length === 0) return null;
    const fit = fitRegressionModel(trainSet, modelType, params);
    const probs = evalSet.map((point) => Math.min(0.999, Math.max(0.001, fit.predict(point.features))));
    const truth = evalSet.map((point) => point.y >= 0.5 ? 1 : 0);
    const binCount = 8;
    const bins = Array.from({ length: binCount }, (_, index) => ({
      low: index / binCount,
      high: (index + 1) / binCount,
      avgProb: 0,
      fracPos: 0,
      count: 0,
    }));
    for (let i = 0; i < probs.length; i++) {
      const p = probs[i];
      const b = Math.min(binCount - 1, Math.floor(p * binCount));
      bins[b].avgProb += p;
      bins[b].fracPos += truth[i];
      bins[b].count += 1;
    }
    const calib = bins
      .filter((bin) => bin.count > 0)
      .map((bin) => ({
        x: bin.avgProb / bin.count,
        y: bin.fracPos / bin.count,
        n: bin.count,
      }));

    const thresholds = Array.from({ length: 19 }, (_, i) => Number((0.05 + i * 0.05).toFixed(2)));
    const sens = thresholds.map((t) => {
      let tp = 0;
      let fp = 0;
      let fn = 0;
      for (let i = 0; i < probs.length; i++) {
        const pred = probs[i] >= t ? 1 : 0;
        if (pred === 1 && truth[i] === 1) tp += 1;
        if (pred === 1 && truth[i] === 0) fp += 1;
        if (pred === 0 && truth[i] === 1) fn += 1;
      }
      const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
      const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
      const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
      return { threshold: t, f1, precision, recall };
    });

    return { calibration: calib, sensitivity: sens };
  }, [data, evaluationMode, split, modelType, params]);

  if (!diagnostic) return null;
  const [row0, row1] = diagnostic.confusion;
  const total = row0[0] + row0[1] + row1[0] + row1[1];
  const actual0Total = row0[0] + row0[1];
  const actual1Total = row1[0] + row1[1];
  const pred0Total = row0[0] + row1[0];
  const pred1Total = row0[1] + row1[1];
  const confAnnotations = [
    { x: 'Pred 0', y: 'Actual 0', value: diagnostic.confusion[0][0] },
    { x: 'Pred 1', y: 'Actual 0', value: diagnostic.confusion[0][1] },
    { x: 'Pred 0', y: 'Actual 1', value: diagnostic.confusion[1][0] },
    { x: 'Pred 1', y: 'Actual 1', value: diagnostic.confusion[1][1] },
  ];

  return (
    <div className={compact ? 'space-y-2.5' : 'space-y-3'}>
      <div className="flex items-center justify-between">
        <p className="text-xs text-text-secondary">
          Diagnostics evaluate {evaluationMode === 'train_test' ? 'held-out' : 'current'} samples at threshold {threshold.toFixed(2)}.
        </p>
        <button
          type="button"
          className={`eval-chip !px-2.5 !py-1 ${showConfidenceView ? 'eval-chip-active' : ''}`}
          onClick={() => setShowConfidenceView((value) => !value)}
        >
          Confidence View
        </button>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-2.5">
        <div className="plot-wrap code-block h-64 overflow-hidden">
          <Plot
            key={`diag-conf-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
            data={[{
              x: ['Pred 0', 'Pred 1'],
              y: ['Actual 0', 'Actual 1'],
              z: diagnostic.confusion,
              type: 'heatmap',
              colorscale: isDark
                ? [
                    [0, '#334155'],
                    [0.5, '#475569'],
                    [1, '#64748b'],
                  ]
                : [
                    [0, '#f8fafc'],
                    [0.5, '#bfdbfe'],
                    [1, '#60a5fa'],
                  ],
              showscale: false,
              hovertemplate: '%{y} / %{x}: %{z}<extra></extra>',
            }]}
            layout={{
              margin: { l: 54, r: 12, t: 34, b: 36 },
              title: { text: 'Confusion Matrix', font: { size: 12, color: axisText } },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: axisText, family: 'Inter, sans-serif' },
              annotations: confAnnotations.map((entry) => ({
                x: entry.x,
                y: entry.y,
                text: `${entry.value}`,
                showarrow: false,
                font: {
                  size: 14,
                  color: isDark ? '#f8fafc' : '#0f172a',
                },
              })),
              xaxis: { color: axisText, gridcolor: grid, zerolinecolor: zero },
              yaxis: { color: axisText, gridcolor: grid, zerolinecolor: zero },
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
          <div className="mt-1.5 grid grid-cols-2 gap-1 text-[11px] text-text-secondary">
            <span className="status-pill">Total: {total}</span>
            <span className="status-pill">Actual 0: {actual0Total}</span>
            <span className="status-pill">Actual 1: {actual1Total}</span>
            <span className="status-pill">Pred 0: {pred0Total} | Pred 1: {pred1Total}</span>
          </div>
        </div>
        <div className="plot-wrap code-block h-64 overflow-hidden">
          <Plot
            key={`diag-roc-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
            data={[
              {
                x: diagnostic.rocCurve.map((point) => point.fpr),
                y: diagnostic.rocCurve.map((point) => point.tpr),
                type: 'scatter',
                mode: 'lines',
                name: `ROC AUC ${diagnostic.rocAuc.toFixed(3)}`,
                line: { color: '#2563eb', width: 2.2 },
              },
              {
                x: [0, 1],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                showlegend: false,
                line: { color: isDark ? 'rgba(148,163,184,0.45)' : 'rgba(100,116,139,0.45)', width: 1, dash: 'dot' },
                hoverinfo: 'skip',
              },
            ]}
            layout={{
              margin: { l: 48, r: 12, t: 34, b: 36 },
              title: { text: 'ROC Curve', font: { size: 12, color: axisText } },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: axisText, family: 'Inter, sans-serif' },
              xaxis: { title: { text: 'FPR' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
              yaxis: { title: { text: 'TPR' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
              legend: { x: 0.03, y: 0.97, bgcolor: legendBg, bordercolor: legendBorder, borderwidth: 1, font: { size: 11, color: axisText } },
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
        <div className="plot-wrap code-block h-64 overflow-hidden">
          <Plot
            key={`diag-pr-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
            data={[{
              x: diagnostic.prCurve.map((point) => point.recall),
              y: diagnostic.prCurve.map((point) => point.precision),
              type: 'scatter',
              mode: 'lines',
              name: `PR AUC ${diagnostic.prAuc.toFixed(3)}`,
              line: { color: '#f59e0b', width: 2.2 },
            }]}
            layout={{
              margin: { l: 48, r: 12, t: 34, b: 36 },
              title: { text: 'Precision-Recall Curve', font: { size: 12, color: axisText } },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: axisText, family: 'Inter, sans-serif' },
              xaxis: { title: { text: 'Recall' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
              yaxis: { title: { text: 'Precision' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
              legend: { x: 0.03, y: 0.97, bgcolor: legendBg, bordercolor: legendBorder, borderwidth: 1, font: { size: 11, color: axisText } },
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
      {featureFlags.ff_info_microcards && (
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>Color: counts and curve quality across thresholds.</span>
          <InfoPopover
            label="Classification Diagnostics"
            what="Confusion matrix gives threshold-specific errors; ROC/PR summarize threshold sweeps."
            why="This separates boundary placement from ranking quality."
            tryNext="Shift threshold and compare confusion changes while ROC/PR stay threshold-agnostic."
          />
        </div>
      )}
      {showConfidenceView && confidenceDiagnostics && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-2.5">
          <div className="plot-wrap code-block h-56 overflow-hidden">
            <Plot
              data={[
                {
                  x: confidenceDiagnostics.calibration.map((point) => point.x),
                  y: confidenceDiagnostics.calibration.map((point) => point.y),
                  mode: 'lines+markers',
                  type: 'scatter',
                  marker: { size: confidenceDiagnostics.calibration.map((point) => Math.max(7, point.n + 4)), color: '#0ea5e9', opacity: 0.8 },
                  line: { color: '#0284c7', width: 2 },
                  name: 'Calibration bins',
                },
                {
                  x: [0, 1],
                  y: [0, 1],
                  type: 'scatter',
                  mode: 'lines',
                  showlegend: false,
                  line: { color: isDark ? 'rgba(148,163,184,0.45)' : 'rgba(100,116,139,0.45)', width: 1, dash: 'dot' },
                  hoverinfo: 'skip',
                },
              ]}
              layout={{
                margin: { l: 46, r: 12, t: 28, b: 38 },
                title: { text: 'Calibration Bins', font: { size: 12, color: axisText } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: axisText, family: 'Inter, sans-serif' },
                xaxis: { title: { text: 'Predicted probability' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
                yaxis: { title: { text: 'Observed positive rate' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
              }}
              config={{ responsive: true, displayModeBar: false }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
          <div className="plot-wrap code-block h-56 overflow-hidden">
            <Plot
              data={[
                {
                  x: confidenceDiagnostics.sensitivity.map((point) => point.threshold),
                  y: confidenceDiagnostics.sensitivity.map((point) => point.f1),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'F1',
                  line: { color: '#10b981', width: 2 },
                },
                {
                  x: confidenceDiagnostics.sensitivity.map((point) => point.threshold),
                  y: confidenceDiagnostics.sensitivity.map((point) => point.precision),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Precision',
                  line: { color: '#f59e0b', width: 1.8 },
                },
                {
                  x: confidenceDiagnostics.sensitivity.map((point) => point.threshold),
                  y: confidenceDiagnostics.sensitivity.map((point) => point.recall),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Recall',
                  line: { color: '#2563eb', width: 1.8 },
                },
              ]}
              layout={{
                margin: { l: 46, r: 12, t: 28, b: 38 },
                title: { text: 'Threshold Sensitivity', font: { size: 12, color: axisText } },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: axisText, family: 'Inter, sans-serif' },
                xaxis: { title: { text: 'Decision threshold' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0.05, 0.95] },
                yaxis: { title: { text: 'Score' }, color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
                legend: { x: 0.02, y: 0.98, bgcolor: legendBg, bordercolor: legendBorder, borderwidth: 1, font: { size: 10, color: axisText } },
              }}
              config={{ responsive: true, displayModeBar: false }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
