import Plot from 'react-plotly.js';
import { useModelStore } from '../../store/modelStore';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { fitRegressionModel } from '../../lib/dataUtils';
import { useTheme } from 'next-themes';
import { InfoPopover } from '../InfoPopover';
import { featureFlags } from '../../config/featureFlags';

interface AssumptionCheckerProps {
  sidebarCollapsed?: boolean;
  compact?: boolean;
}

export function AssumptionChecker({ sidebarCollapsed = false, compact = false }: AssumptionCheckerProps) {
  const { resolvedTheme } = useTheme();
  const { data, params, modelType } = useModelStore();
  const isDark = resolvedTheme === 'dark';
  const axisText = isDark ? '#94a3b8' : '#475569';
  const grid = isDark ? 'rgba(100,116,139,0.22)' : 'rgba(148,163,184,0.35)';
  const hoverBg = isDark ? 'rgba(30,41,59,0.78)' : 'rgba(255,255,255,0.84)';
  const hoverBorder = isDark ? 'rgba(100,116,139,0.5)' : 'rgba(148,163,184,0.55)';

  if (data.length === 0) return null;
  const fittedModel = fitRegressionModel(data, modelType, params);

  // Compute residuals
  const residuals = data.map((p) => ({
    x: p.x,
    fitted: fittedModel.predict(p.features),
    residual: p.y - fittedModel.predict(p.features),
  }));

  const fittedValues = residuals.map((r) => r.fitted);
  const residualValues = residuals.map((r) => r.residual);

  // Q-Q plot data (simplified)
  const sortedResiduals = [...residualValues].sort((a, b) => a - b);
  const n = sortedResiduals.length;
  
  // Approximate inverse normal CDF using simple method
  const theoreticalQuantiles = sortedResiduals.map((_, i) => {
    const p = (i + 0.5) / n;
    // Simple approximation of inverse normal (probit function)
    if (p < 0.5) {
      return -Math.sqrt(-2 * Math.log(p));
    } else {
      return Math.sqrt(-2 * Math.log(1 - p));
    }
  });

  // Check assumptions
  const meanResidual = residualValues.reduce((a, b) => a + b, 0) / residualValues.length;
  const isMeanZero = Math.abs(meanResidual) < 0.1;

  // Simple heteroscedasticity check (correlation between |residual| and fitted)
  const absResiduals = residualValues.map(Math.abs);
  const fittedMean = fittedValues.reduce((a, b) => a + b, 0) / fittedValues.length;
  const absResidMean = absResiduals.reduce((a, b) => a + b, 0) / absResiduals.length;
  
  let numerator = 0;
  let denomFitted = 0;
  let denomAbsResid = 0;
  
  for (let i = 0; i < fittedValues.length; i++) {
    const diffFitted = fittedValues[i] - fittedMean;
    const diffAbsResid = absResiduals[i] - absResidMean;
    numerator += diffFitted * diffAbsResid;
    denomFitted += diffFitted * diffFitted;
    denomAbsResid += diffAbsResid * diffAbsResid;
  }
  
  const correlation = numerator / Math.sqrt(denomFitted * denomAbsResid);
  const isHomoscedastic = Math.abs(correlation) < 0.3;

  return (
    <div className={compact ? 'space-y-3' : 'space-y-4'}>
      <div className="flex items-center justify-between">
        <h3 className="panel-title">
          Model Diagnostics
        </h3>
        <div className="flex gap-4">
          <div className="flex items-center gap-1.5">
            {isMeanZero ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-amber-500" />
            )}
            <span className={`text-xs ${isMeanZero ? 'text-emerald-400' : 'text-amber-400'}`}>
              Zero Mean
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            {isHomoscedastic ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-amber-500" />
            )}
            <span className={`text-xs ${isHomoscedastic ? 'text-emerald-400' : 'text-amber-400'}`}>
              Homoscedasticity
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Residuals vs Fitted */}
        <div className="plot-wrap code-block h-64 overflow-hidden">
          <Plot
            key={`residuals-fit-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
            data={[
              {
                x: fittedValues,
                y: residualValues,
                mode: 'markers',
                type: 'scatter',
                marker: { color: '#94a3b8', size: 8 },
                name: 'Residuals',
              },
              {
                x: [Math.min(...fittedValues), Math.max(...fittedValues)],
                y: [0, 0],
                mode: 'lines',
                type: 'scatter',
                line: { color: '#ef4444', width: 2, dash: 'dash' },
                name: 'Zero Line',
              },
            ]}
            layout={{
              title: { text: 'Residuals vs Fitted', font: { size: 12, color: axisText } },
              margin: { l: 40, r: 20, t: 30, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: axisText, size: 10 },
              xaxis: { 
                title: { text: 'Fitted Values', standoff: 8, font: { size: 11, color: axisText } },
                color: axisText,
                tickfont: { size: 10 },
                gridcolor: grid,
              },
              yaxis: { 
                title: { text: 'Residuals', standoff: 8, font: { size: 11, color: axisText } },
                color: axisText,
                tickfont: { size: 10 },
                gridcolor: grid,
              },
              hoverlabel: { bgcolor: hoverBg, bordercolor: hoverBorder, font: { color: axisText, size: 10 } },
              showlegend: false,
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>

        {/* Q-Q Plot */}
        <div className="plot-wrap code-block h-64 overflow-hidden">
          <Plot
            key={`qq-plot-${sidebarCollapsed ? 'collapsed' : 'expanded'}`}
            data={[
              {
                x: theoreticalQuantiles,
                y: sortedResiduals,
                mode: 'markers',
                type: 'scatter',
                marker: { color: '#94a3b8', size: 8 },
                name: 'Q-Q',
              },
              {
                x: [Math.min(...theoreticalQuantiles), Math.max(...theoreticalQuantiles)],
                y: [Math.min(...theoreticalQuantiles), Math.max(...theoreticalQuantiles)],
                mode: 'lines',
                type: 'scatter',
                line: { color: '#ef4444', width: 2, dash: 'dash' },
                name: 'Reference',
              },
            ]}
            layout={{
              title: { text: 'Normal Q-Q Plot', font: { size: 12, color: axisText } },
              margin: { l: 40, r: 20, t: 30, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: axisText, size: 10 },
              xaxis: { 
                title: { text: 'Theoretical Quantiles', standoff: 8, font: { size: 11, color: axisText } },
                color: axisText,
                tickfont: { size: 10 },
                gridcolor: grid,
              },
              yaxis: { 
                title: { text: 'Sample Quantiles', standoff: 8, font: { size: 11, color: axisText } },
                color: axisText,
                tickfont: { size: 10 },
                gridcolor: grid,
              },
              hoverlabel: { bgcolor: hoverBg, bordercolor: hoverBorder, font: { color: axisText, size: 10 } },
              showlegend: false,
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>

      <div className="text-xs text-text-secondary space-y-1">
        <p className="inline-flex items-start gap-1.5">
          <strong className="text-text-primary">Residuals vs Fitted:</strong>
          Random horizontal spread supports linearity and constant variance; fan-shapes indicate heteroscedasticity or missing nonlinear terms.
          {featureFlags.ff_info_microcards && (
            <InfoPopover
              label="Residuals vs Fitted"
              what="This plot compares residual size against fitted values."
              why="Patterns reveal misspecification, nonlinearity, or variance drift."
              tryNext="Switch to heteroscedastic data, then compare Ridge vs OLS."
            />
          )}
        </p>
        <p className="inline-flex items-start gap-1.5">
          <strong className="text-text-primary">Q-Q Plot:</strong>
          Alignment to the 45° line indicates approximately normal residuals; heavy tail deviations indicate non-Gaussian error structure.
          {featureFlags.ff_info_microcards && (
            <InfoPopover
              label="Normal Q-Q Plot"
              what="Residual quantiles are compared with normal-theory quantiles."
              why="Strong deviations suggest heavy tails or skewed error structure."
              tryNext="Use outliers dataset and inspect tail deviations after resampling."
            />
          )}
        </p>
      </div>
    </div>
  );
}
