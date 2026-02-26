import Plot from 'react-plotly.js';
import { useMemo, useState } from 'react';
import { useModelStore } from '../store/modelStore';
import { fitRegressionModel, splitDataset } from '../lib/dataUtils';
import { useTheme } from 'next-themes';

interface VisualizerProps {
  sidebarCollapsed?: boolean;
}

export function Visualizer({ sidebarCollapsed = false }: VisualizerProps) {
  const [cameraPreset, setCameraPreset] = useState<'front' | 'iso' | 'top'>('iso');
  const { resolvedTheme } = useTheme();
  const { data, taskMode, params, showOlsSolution, modelType, evaluationMode, testRatio, randomSeed, datasetVersion } = useModelStore();
  const isDark = resolvedTheme === 'dark';
  const axisText = isDark ? '#94a3b8' : '#475569';
  const grid = isDark ? 'rgba(100,116,139,0.24)' : 'rgba(148,163,184,0.35)';
  const zero = isDark ? 'rgba(148,163,184,0.35)' : 'rgba(100,116,139,0.35)';
  const legendBg = isDark ? 'rgba(30,41,59,0.78)' : 'rgba(255,255,255,0.82)';
  const legendBorder = isDark ? 'rgba(100,116,139,0.45)' : 'rgba(148,163,184,0.5)';

  const split = useMemo(
    () => (evaluationMode === 'train_test'
      ? splitDataset(data, testRatio, randomSeed + datasetVersion, taskMode === 'classification')
      : { train: data, test: [] as typeof data }),
    [evaluationMode, data, testRatio, randomSeed, datasetVersion, taskMode]
  );
  const activeFit = useMemo(
    () => fitRegressionModel(taskMode === 'classification' && evaluationMode === 'train_test' ? split.train : data, modelType, params),
    [taskMode, evaluationMode, split.train, data, modelType, params]
  );
  const olsFit = useMemo(
    () => (showOlsSolution ? fitRegressionModel(data, 'ols', params) : null),
    [showOlsSolution, data, params]
  );

  if (data.length === 0) {
    return (
      <div className="plot-wrap code-block h-96 flex items-center justify-center">
        <p className="text-text-tertiary">Loading data...</p>
      </div>
    );
  }

  const isMultiFeature = data.some((point) => typeof point.x2 === 'number');
  const xVals = data.map((point) => point.x);
  const minX = Math.min(...xVals);
  const maxX = Math.max(...xVals);
  const gridPoints = 120;
  const lineX = Array.from({ length: gridPoints }, (_, index) => minX + ((maxX - minX) * index) / (gridPoints - 1));
  const activeY = lineX.map((x) => activeFit.predict(x));
  const olsY = olsFit ? lineX.map((x) => olsFit.predict(x)) : null;
  if (taskMode === 'classification') {
    const x1Vals = data.map((point) => point.features[0]);
    const x2Vals = data.map((point) => point.features[1] ?? 0);
    const minX1 = Math.min(...x1Vals);
    const maxX1 = Math.max(...x1Vals);
    const minX2 = Math.min(...x2Vals);
    const maxX2 = Math.max(...x2Vals);
    const fit = activeFit;
    const threshold = Math.min(0.95, Math.max(0.05, params.decisionThreshold));
    const gridSize = data.length > 320 ? 34 : data.length > 220 ? 40 : 48;
    const regionX = Array.from({ length: gridSize }, (_, i) => minX1 + ((maxX1 - minX1) * i) / (gridSize - 1));
    const regionY = Array.from({ length: gridSize }, (_, i) => minX2 + ((maxX2 - minX2) * i) / (gridSize - 1));
    const regionZ = regionY.map((y) => regionX.map((x) => fit.predict([x, y])));

    const activeEval = evaluationMode === 'train_test' ? split.test : split.train;
    const backgroundTrain = evaluationMode === 'train_test' ? split.train : [];
    const class0Correct = { x: [] as number[], y: [] as number[] };
    const class0Wrong = { x: [] as number[], y: [] as number[] };
    const class1Correct = { x: [] as number[], y: [] as number[] };
    const class1Wrong = { x: [] as number[], y: [] as number[] };
    let tp = 0;
    let tn = 0;
    let fp = 0;
    let fn = 0;
    for (const point of activeEval) {
      const truth = point.y >= 0.5 ? 1 : 0;
      const pred = fit.predict(point.features) >= threshold ? 1 : 0;
      if (truth === 0 && pred === 0) {
        class0Correct.x.push(point.features[0]);
        class0Correct.y.push(point.features[1] ?? 0);
      } else if (truth === 0 && pred === 1) {
        class0Wrong.x.push(point.features[0]);
        class0Wrong.y.push(point.features[1] ?? 0);
      } else if (truth === 1 && pred === 1) {
        class1Correct.x.push(point.features[0]);
        class1Correct.y.push(point.features[1] ?? 0);
      } else {
        class1Wrong.x.push(point.features[0]);
        class1Wrong.y.push(point.features[1] ?? 0);
      }
      if (truth === 1 && pred === 1) tp += 1;
      if (truth === 0 && pred === 0) tn += 1;
      if (truth === 0 && pred === 1) fp += 1;
      if (truth === 1 && pred === 0) fn += 1;
    }

    const marginDelta = 0.14;
    const lowerMargin = Math.max(0.02, threshold - marginDelta);
    const upperMargin = Math.min(0.98, threshold + marginDelta);
    const evalLabel = evaluationMode === 'train_test' ? 'Held-out' : 'Evaluated';

    return (
      <div className="space-y-2">
        <div className="plot-wrap code-block h-[clamp(320px,52vh,560px)] overflow-hidden">
          <Plot
            key={`plot-cls-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
            data={[
              {
                x: regionX,
                y: regionY,
                z: regionZ,
                type: 'contour',
                contours: { start: 0, end: 1, size: 1, coloring: 'fill', showlines: false },
                colorscale: [
                  [0, isDark ? 'rgba(59,130,246,0.18)' : 'rgba(37,99,235,0.16)'],
                  [0.499, isDark ? 'rgba(59,130,246,0.18)' : 'rgba(37,99,235,0.16)'],
                  [0.5, isDark ? 'rgba(244,63,94,0.18)' : 'rgba(220,38,38,0.14)'],
                  [1, isDark ? 'rgba(244,63,94,0.18)' : 'rgba(220,38,38,0.14)'],
                ],
                opacity: 0.35,
                showscale: false,
                name: 'Predicted Region',
                hoverinfo: 'skip',
              },
              {
                x: regionX,
                y: regionY,
                z: regionZ,
                type: 'contour',
                contours: { start: threshold, end: threshold, size: 1, coloring: 'lines', showlabels: false },
                line: { color: isDark ? '#f8fafc' : '#0f172a', width: 2.2 },
                showscale: false,
                name: 'Decision Boundary',
                hoverinfo: 'skip',
              },
              {
                x: regionX,
                y: regionY,
                z: regionZ,
                type: 'contour',
                contours: { start: lowerMargin, end: lowerMargin, size: 1, coloring: 'lines', showlabels: false },
                line: { color: isDark ? 'rgba(248,250,252,0.5)' : 'rgba(15,23,42,0.45)', width: 1, dash: 'dot' },
                showscale: false,
                name: 'Uncertain Band',
                hoverinfo: 'skip',
              },
              {
                x: regionX,
                y: regionY,
                z: regionZ,
                type: 'contour',
                contours: { start: upperMargin, end: upperMargin, size: 1, coloring: 'lines', showlabels: false },
                line: { color: isDark ? 'rgba(248,250,252,0.5)' : 'rgba(15,23,42,0.45)', width: 1, dash: 'dot' },
                showscale: false,
                showlegend: false,
                hoverinfo: 'skip',
              },
              ...(backgroundTrain.length > 0
                ? [{
                  x: backgroundTrain.map((point) => point.features[0]),
                  y: backgroundTrain.map((point) => point.features[1] ?? 0),
                  mode: 'markers',
                  type: 'scatter',
                  marker: {
                    color: isDark ? 'rgba(148,163,184,0.35)' : 'rgba(100,116,139,0.28)',
                    size: 6,
                    symbol: 'circle',
                  },
                  name: 'Train Context',
                }]
                : []),
              {
                x: class0Correct.x,
                y: class0Correct.y,
                mode: 'markers',
                type: 'scatter',
                marker: { color: isDark ? '#60a5fa' : '#2563eb', size: 9, symbol: 'circle' },
                name: `Class 0 ${evalLabel} Correct`,
              },
              {
                x: class1Correct.x,
                y: class1Correct.y,
                mode: 'markers',
                type: 'scatter',
                marker: { color: isDark ? '#f87171' : '#dc2626', size: 9, symbol: 'circle' },
                name: `Class 1 ${evalLabel} Correct`,
              },
              {
                x: class0Wrong.x,
                y: class0Wrong.y,
                mode: 'markers',
                type: 'scatter',
                marker: {
                  color: isDark ? '#60a5fa' : '#2563eb',
                  size: 11,
                  symbol: 'x',
                  line: { color: isDark ? '#bfdbfe' : '#1e3a8a', width: 1 },
                },
                name: `Class 0 ${evalLabel} Wrong`,
              },
              {
                x: class1Wrong.x,
                y: class1Wrong.y,
                mode: 'markers',
                type: 'scatter',
                marker: {
                  color: isDark ? '#f87171' : '#dc2626',
                  size: 11,
                  symbol: 'x',
                  line: { color: isDark ? '#fecaca' : '#7f1d1d', width: 1 },
                },
                name: `Class 1 ${evalLabel} Wrong`,
              },
            ] as any[]}
            layout={{
              margin: { l: 44, r: 12, t: 16, b: 46 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              dragmode: 'pan',
              font: { color: 'hsl(var(--text-secondary))', family: 'Inter, sans-serif' },
              xaxis: { title: { text: 'Feature 1' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
              yaxis: { title: { text: 'Feature 2' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
              legend: { x: 0.02, y: 0.98, bgcolor: legendBg, bordercolor: legendBorder, borderwidth: 1, font: { size: 11, color: axisText } },
              annotations: [{
                xref: 'paper',
                yref: 'paper',
                x: 0.995,
                y: 1.08,
                xanchor: 'right',
                yanchor: 'bottom',
                text: `threshold ${threshold.toFixed(2)} · ${evalLabel.toLowerCase()} n=${activeEval.length}`,
                showarrow: false,
                font: { size: 11, color: axisText },
              }, {
                xref: 'paper',
                yref: 'paper',
                x: 0.01,
                y: -0.12,
                xanchor: 'left',
                yanchor: 'top',
                text: 'Color = true class · Circle = correct · X = misclassified',
                showarrow: false,
                font: { size: 10, color: axisText },
              }],
            }}
            config={{ responsive: true, displayModeBar: false, scrollZoom: true, doubleClick: 'reset+autosize' }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <div className="metric-card p-2 text-xs"><span className="text-text-tertiary">True Positives <span className="text-[10px]">(TP)</span></span><div className="text-text-primary font-semibold">{tp}</div></div>
          <div className="metric-card p-2 text-xs"><span className="text-text-tertiary">False Positives <span className="text-[10px]">(FP)</span></span><div className="text-text-primary font-semibold">{fp}</div></div>
          <div className="metric-card p-2 text-xs"><span className="text-text-tertiary">True Negatives <span className="text-[10px]">(TN)</span></span><div className="text-text-primary font-semibold">{tn}</div></div>
          <div className="metric-card p-2 text-xs"><span className="text-text-tertiary">False Negatives <span className="text-[10px]">(FN)</span></span><div className="text-text-primary font-semibold">{fn}</div></div>
        </div>
      </div>
    );
  }

  const residualSigma = Math.sqrt(
    split.train.reduce((sum, point) => {
      const residual = point.y - activeFit.predict(point.features);
      return sum + residual * residual;
    }, 0) / Math.max(split.train.length, 1)
  );
  const bandUpper = lineX.map((_, index) => activeY[index] + residualSigma);
  const bandLower = lineX.map((_, index) => activeY[index] - residualSigma);

  if (isMultiFeature) {
    const x1Vals = data.map((point) => point.features[0]);
    const x2Vals = data.map((point) => point.features[1] ?? 0);
    const minX1 = Math.min(...x1Vals);
    const maxX1 = Math.max(...x1Vals);
    const minX2 = Math.min(...x2Vals);
    const maxX2 = Math.max(...x2Vals);
    const gridSize = 18;
    const surfaceX = Array.from({ length: gridSize }, (_, i) => minX1 + ((maxX1 - minX1) * i) / (gridSize - 1));
    const surfaceY = Array.from({ length: gridSize }, (_, i) => minX2 + ((maxX2 - minX2) * i) / (gridSize - 1));
    const surfaceZ = surfaceY.map((fy) => surfaceX.map((fx) => activeFit.predict([fx, fy])));

    const threeDData: any[] = [
      {
        x: surfaceX,
        y: surfaceY,
        z: surfaceZ,
        type: 'surface',
        opacity: 0.5,
        showscale: false,
        colorscale: isDark ? 'Blues' : 'Bluered',
        name: 'Model Surface',
      },
      {
        x: split.train.map((point) => point.features[0]),
        y: split.train.map((point) => point.features[1] ?? 0),
        z: split.train.map((point) => point.y),
        mode: 'markers',
        type: 'scatter3d',
        marker: { size: 4, color: isDark ? '#93c5fd' : '#1d4ed8', opacity: 0.85 },
        name: evaluationMode === 'train_test' ? 'Train Points' : 'Data Points',
      },
    ];
    if (evaluationMode === 'train_test') {
      threeDData.push({
        x: split.test.map((point) => point.features[0]),
        y: split.test.map((point) => point.features[1] ?? 0),
        z: split.test.map((point) => point.y),
        mode: 'markers',
        type: 'scatter3d',
        marker: { size: 5, color: isDark ? '#fbbf24' : '#b45309', symbol: 'diamond', opacity: 0.95 },
        name: 'Test Points',
      });
    }

    const cameraByPreset = {
      front: { eye: { x: 0.01, y: 2.2, z: 0.6 }, up: { x: 0, y: 0, z: 1 } },
      iso: { eye: { x: 1.3, y: 1.25, z: 0.85 }, up: { x: 0, y: 0, z: 1 } },
      top: { eye: { x: 0.01, y: 0.01, z: 2.45 }, up: { x: 0, y: 1, z: 0 } },
    } as const;

    return (
      <div className="plot-wrap code-block h-[clamp(340px,58vh,620px)] overflow-hidden">
        <div className="absolute top-2 right-2 z-10 flex gap-1">
          <button type="button" className={`quick-action ${cameraPreset === 'front' ? 'quick-action-active' : ''}`} onClick={() => setCameraPreset('front')}>Front</button>
          <button type="button" className={`quick-action ${cameraPreset === 'iso' ? 'quick-action-active' : ''}`} onClick={() => setCameraPreset('iso')}>Iso</button>
          <button type="button" className={`quick-action ${cameraPreset === 'top' ? 'quick-action-active' : ''}`} onClick={() => setCameraPreset('top')}>Top</button>
        </div>
        <Plot
          key={`plot-3d-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
          data={threeDData}
          layout={{
            margin: { l: 0, r: 0, t: 0, b: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: axisText, family: 'Inter, sans-serif' },
            scene: {
              bgcolor: 'rgba(0,0,0,0)',
              dragmode: 'turntable',
              xaxis: { title: { text: 'Feature 1' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
              yaxis: { title: { text: 'Feature 2' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
              zaxis: { title: { text: 'Target y' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
              camera: cameraByPreset[cameraPreset],
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
          config={{ responsive: true, displayModeBar: false, scrollZoom: true, doubleClick: 'reset+autosize' }}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    );
  }

  const plotData: any[] = [
    {
      x: lineX,
      y: bandLower,
      mode: 'lines',
      type: 'scatter',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      name: 'confidence-lower',
    },
    {
      x: lineX,
      y: bandUpper,
      mode: 'lines',
      type: 'scatter',
      line: { width: 0 },
      fill: 'tonexty',
      fillcolor: isDark ? 'rgba(59,130,246,0.12)' : 'rgba(37,99,235,0.1)',
      hoverinfo: 'skip',
      showlegend: false,
      name: 'Model Uncertainty',
    },
    {
      x: split.train.map((point) => point.x),
      y: split.train.map((point) => point.y),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: isDark ? 'rgba(125, 155, 198, 0.82)' : 'rgba(71, 85, 105, 0.65)',
        size: 9,
        opacity: 0.9,
        line: { color: isDark ? 'rgba(188, 214, 255, 0.35)' : 'rgba(71, 85, 105, 0.24)', width: 0.9 },
      },
      name: evaluationMode === 'train_test' ? 'Train Points' : 'Data Points',
    },
    {
      x: lineX,
      y: activeY,
      mode: 'lines',
      type: 'scatter',
      line: { color: '#2563eb', width: 3 },
      name: `Active Model: ${modelType.split('_').join(' ')}`,
    },
  ];

  if (evaluationMode === 'train_test' && split.test.length > 0) {
    plotData.push({
      x: split.test.map((point) => point.x),
      y: split.test.map((point) => point.y),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: isDark ? 'rgba(251, 191, 36, 0.85)' : 'rgba(217, 119, 6, 0.85)',
        size: 10,
        symbol: 'diamond',
        line: { color: isDark ? 'rgba(255, 220, 138, 0.6)' : 'rgba(146, 64, 14, 0.5)', width: 1 },
      },
      name: 'Test Points',
    });
  }

  if (olsY) {
    plotData.push({
      x: lineX,
      y: olsY,
      mode: 'lines',
      type: 'scatter',
      line: { color: '#10b981', width: 2, dash: 'dash' },
      name: 'OLS Solution',
    });
  }

  return (
    <div className="plot-wrap code-block h-[clamp(320px,52vh,560px)] overflow-hidden">
      <Plot
        key={`plot-2d-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
        data={plotData}
        layout={{
          margin: { l: 44, r: 12, t: 16, b: 46 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          dragmode: 'pan',
          font: { color: 'hsl(var(--text-secondary))', family: 'Inter, sans-serif' },
          xaxis: {
            title: { text: 'X', standoff: 8, font: { size: 12, color: axisText } },
            color: axisText,
            tickfont: { size: 11 },
            gridcolor: grid,
            zerolinecolor: zero,
          },
          yaxis: {
            title: { text: 'Y', standoff: 8, font: { size: 12, color: axisText } },
            color: axisText,
            tickfont: { size: 11 },
            gridcolor: grid,
            zerolinecolor: zero,
          },
          legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: legendBg,
            bordercolor: legendBorder,
            borderwidth: 1,
            font: { size: 11, color: axisText },
          },
          hoverlabel: { bgcolor: legendBg, bordercolor: legendBorder, font: { color: axisText, size: 11 } },
          hovermode: 'x unified',
          transition: { duration: 160, easing: 'cubic-in-out' },
          annotations: [
            {
              xref: 'paper',
              yref: 'paper',
              x: 0.995,
              y: 1.08,
              xanchor: 'right',
              yanchor: 'bottom',
              text:
                evaluationMode === 'train_test'
                  ? `train ${split.train.length} • test ${split.test.length}`
                  : evaluationMode === 'cross_validation'
                    ? 'k-fold validation'
                    : 'full-fit view',
              showarrow: false,
              font: { size: 11, color: axisText },
            },
          ],
        }}
        config={{ responsive: true, displayModeBar: false, scrollZoom: true, doubleClick: 'reset+autosize' }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
