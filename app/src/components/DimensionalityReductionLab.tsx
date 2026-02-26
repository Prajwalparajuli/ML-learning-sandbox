import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useTheme } from 'next-themes';
import { useModelStore } from '../store/modelStore';

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, i) => sum + value * (b[i] ?? 0), 0);
}

function norm(v: number[]): number {
  return Math.sqrt(dot(v, v));
}

function normalize(v: number[]): number[] {
  const n = Math.max(norm(v), 1e-10);
  return v.map((value) => value / n);
}

function matVec(mat: number[][], vec: number[]): number[] {
  return mat.map((row) => dot(row, vec));
}

function covariance(x: number[][]): { cov: number[][]; means: number[] } {
  const p = x[0]?.length ?? 0;
  const means = Array.from({ length: p }, (_, j) => mean(x.map((row) => row[j])));
  const centered = x.map((row) => row.map((value, j) => value - means[j]));
  const cov = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  for (let i = 0; i < centered.length; i++) {
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) cov[a][b] += centered[i][a] * centered[i][b];
    }
  }
  const denom = Math.max(centered.length - 1, 1);
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) cov[a][b] /= denom;
  return { cov, means };
}

function topComponents(x: number[][], k = 2): { components: number[][]; eigenvalues: number[]; means: number[] } {
  const p = x[0]?.length ?? 0;
  const { cov, means } = covariance(x);
  let c = cov.map((row) => [...row]);
  const components: number[][] = [];
  const eigenvalues: number[] = [];
  const count = Math.max(1, Math.min(k, p));
  for (let comp = 0; comp < count; comp++) {
    let v = normalize(Array.from({ length: p }, (_, i) => (i + 1) / (p + 1)));
    for (let it = 0; it < 60; it++) v = normalize(matVec(c, v));
    const lambda = dot(v, matVec(c, v));
    components.push(v);
    eigenvalues.push(Math.max(0, lambda));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) c[i][j] -= lambda * v[i] * v[j];
    }
  }
  return { components, eigenvalues, means };
}

function project(features: number[], means: number[], components: number[][]): number[] {
  const centered = features.map((value, i) => value - (means[i] ?? 0));
  return components.map((component) => dot(centered, component));
}

export function DimensionalityReductionLab() {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === 'dark';
  const axisText = isDark ? '#94a3b8' : '#475569';
  const grid = isDark ? 'rgba(100,116,139,0.24)' : 'rgba(148,163,184,0.35)';
  const zero = isDark ? 'rgba(148,163,184,0.35)' : 'rgba(100,116,139,0.35)';
  const { data, taskMode } = useModelStore();

  const pca = useMemo(() => {
    if (taskMode !== 'regression' || data.length < 4) return null;
    const x = data.map((point) => point.features);
    const y = data.map((point) => point.y);
    const { components, eigenvalues, means } = topComponents(x, 2);
    if (components.length === 0) return null;
    const projected = x.map((row) => project(row, means, components));
    const totalVar = Math.max(eigenvalues.reduce((sum, value) => sum + value, 0), 1e-8);
    const explained = eigenvalues.map((value) => value / totalVar);
    return { projected, explained, y };
  }, [data, taskMode]);

  if (!pca) {
    return (
      <div className="rounded-xl border border-dashed border-border-subtle p-2.5 text-xs text-text-tertiary">
        PCA view appears for regression with at least 4 samples.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-2.5">
      <div className="plot-wrap code-block h-56 overflow-hidden">
        <Plot
          data={[
            {
              x: ['PC1', 'PC2'],
              y: [pca.explained[0] ?? 0, pca.explained[1] ?? 0],
              type: 'bar',
              marker: { color: ['#2563eb', '#0ea5e9'] },
            },
          ]}
          layout={{
            margin: { l: 46, r: 12, t: 30, b: 36 },
            title: { text: 'Explained Variance', font: { size: 12, color: axisText } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: axisText, family: 'Inter, sans-serif' },
            xaxis: { color: axisText, gridcolor: grid, zerolinecolor: zero },
            yaxis: { color: axisText, gridcolor: grid, zerolinecolor: zero, range: [0, 1] },
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
              x: pca.projected.map((row) => row[0] ?? 0),
              y: pca.projected.map((row) => row[1] ?? 0),
              mode: 'markers',
              type: 'scatter',
              marker: {
                size: 7,
                color: pca.y,
                colorscale: isDark ? 'Viridis' : 'Blues',
                showscale: false,
                opacity: 0.88,
              },
              hovertemplate: 'PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra></extra>',
            },
          ]}
          layout={{
            margin: { l: 46, r: 12, t: 30, b: 36 },
            title: { text: 'Component Scatter', font: { size: 12, color: axisText } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: axisText, family: 'Inter, sans-serif' },
            xaxis: { title: { text: 'PC1' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
            yaxis: { title: { text: 'PC2' }, color: axisText, gridcolor: grid, zerolinecolor: zero },
          }}
          config={{ responsive: true, displayModeBar: false }}
          useResizeHandler
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
}

