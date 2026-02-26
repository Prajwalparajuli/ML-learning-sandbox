import { useEffect, useMemo, useState } from 'react';
import { useModelStore } from '../store/modelStore';
import { generateDataset, evaluateModelMetrics, fitRegressionModel, latexForModel } from '../lib/dataUtils';
import { DataControlPanel, ModelControlPanel } from '../components/ControlPanel/ControlPanel';
import { FormulaDisplay } from '../components/FormulaDisplay';
import { Visualizer } from '../components/Visualizer';
import { MetricsBoard } from '../components/MetricsBoard';
import { CodeExporter } from '../components/CodeExporter';
import { AssumptionChecker } from '../components/AssumptionChecker/AssumptionChecker';
import { ClassificationDiagnostics } from '../components/ClassificationDiagnostics';
import { LearningInsights } from '../components/LearningInsights';
import { ClassicalContentPanel } from '../components/ClassicalContentPanel';
import { ScenarioMissions } from '../components/ScenarioMissions';
import { InfoPopover } from '../components/InfoPopover';
import { Toaster } from '../components/ui/sonner';
import { toast } from 'sonner';
import { Compass, Sparkles, PlayCircle, TestTube2, Layers2, Shuffle, PanelLeftClose, PanelLeftOpen, BookOpen, Route, Target } from 'lucide-react';
import { ThemeToggle } from '../components/ThemeToggle';
import { datasetExplanations, modelContentMap } from '../content/classicalContentAdapter';
import { featureFlags } from '../config/featureFlags';

export function InteractiveModelTemplate() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [focusPanel, setFocusPanel] = useState<'none' | 'lens' | 'diagnostics' | 'playbook' | 'missions'>('none');
  const [showEquation, setShowEquation] = useState(false);
  const {
    taskMode,
    setTaskMode,
    modelType,
    dataset,
    params,
    data,
    sampleSize,
    randomSeed,
    datasetVersion,
    evaluationMode,
    featureMode,
    testRatio,
    cvFolds,
    showAssumptions,
    showClassificationDiagnostics,
    showOlsSolution,
    compareWithOls,
    heroLayoutMode,
    setHeroLayoutMode,
    viewMode,
    setViewMode,
    setData,
    setMetrics,
    error,
    setError,
    setShowAssumptions,
    setShowClassificationDiagnostics,
    setShowOlsSolution,
    setCompareWithOls,
    regenerateDataset,
  } = useModelStore();

  // Generate data when dataset changes
  useEffect(() => {
    try {
      const newData = generateDataset(dataset, sampleSize, randomSeed + datasetVersion, featureMode);
      setData(newData);
      setError(null);
    } catch {
      setError('Failed to generate dataset. Please try again.');
    }
  }, [dataset, sampleSize, randomSeed, datasetVersion, featureMode, setData, setError]);

  // Compute metrics when parameters or data change
  useEffect(() => {
    if (data.length === 0) return;

    const timeout = window.setTimeout(() => {
      try {
        const newMetrics = evaluateModelMetrics(
          data,
          modelType,
          params,
          evaluationMode,
          testRatio,
          cvFolds,
          randomSeed + datasetVersion
        );
        setMetrics(newMetrics);
        setError(null);
      } catch {
        setError('Computation error: Check your parameters');
      }
    }, 90);
    return () => window.clearTimeout(timeout);
  }, [params, data, modelType, evaluationMode, testRatio, cvFolds, randomSeed, datasetVersion, setMetrics, setError]);

  // Show error toast when error changes
  useEffect(() => {
    if (error) {
      toast.error(error, { id: 'sandbox-error' });
    }
  }, [error]);

  useEffect(() => {
    if (viewMode === 'deep_dive') {
      setFocusPanel('none');
    }
  }, [viewMode]);

  const fittedForFormula = useMemo(
    () => (data.length > 0 ? fitRegressionModel(data, modelType, params) : null),
    [data, modelType, params]
  );
  const currentFormula = latexForModel(modelType, params, fittedForFormula);

  const renderDiagnosticsSection = (compact = false, forceVisible = false) => {
    if (taskMode === 'regression' && (showAssumptions || forceVisible)) {
      return (
        <section className={`material-panel ${compact ? 'p-2.5' : 'p-3'} motion-stagger`} style={{ ['--stagger-index' as any]: 2 }}>
          <AssumptionChecker sidebarCollapsed={sidebarCollapsed} compact={compact} />
        </section>
      );
    }
    if (taskMode === 'classification' && (showClassificationDiagnostics || forceVisible)) {
      return (
        <section className={`material-panel ${compact ? 'p-2.5' : 'p-3'} motion-stagger`} style={{ ['--stagger-index' as any]: 2 }}>
          <ClassificationDiagnostics sidebarCollapsed={sidebarCollapsed} compact={compact} />
        </section>
      );
    }
    return null;
  };

  return (
    <div className="app-canvas min-h-screen">
      <Toaster position="top-right" richColors />
      
      {/* Header */}
      <header className="app-header sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-3 sm:px-5 py-1.5">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-apple-sm bg-accent/12 border border-border-subtle flex items-center justify-center">
                <svg className="w-4.5 h-4.5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-base sm:text-lg leading-none font-semibold text-text-primary">ML Learning Sandbox</h1>
                <p className="hidden md:block text-xs text-text-secondary mt-0.5">Inspect, tune, compare, and export in real time</p>
              </div>
            </div>
            <nav className="w-full sm:w-auto flex flex-wrap sm:flex-nowrap items-center gap-1.5 sm:gap-2">
              <div className="theme-toggle-shell">
                <button
                  type="button"
                  className={`theme-toggle-chip ${taskMode === 'regression' ? 'theme-toggle-chip-active' : ''}`}
                  onClick={() => setTaskMode('regression')}
                >
                  Regression
                </button>
                <button
                  type="button"
                  className={`theme-toggle-chip ${taskMode === 'classification' ? 'theme-toggle-chip-active' : ''}`}
                  onClick={() => setTaskMode('classification')}
                >
                  Classification
                </button>
              </div>
              <ThemeToggle />
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full px-3 lg:pl-0 lg:pr-4 py-3">
        <div className={`grid grid-cols-1 gap-4 lg:transition-[grid-template-columns] lg:duration-300 lg:ease-out ${sidebarCollapsed ? 'lg:grid-cols-[68px_minmax(0,1fr)_minmax(250px,18%)]' : 'lg:grid-cols-[minmax(250px,18%)_minmax(0,1fr)_minmax(250px,18%)]'}`}>
          {/* Left Pane: Data Sidebar */}
          <aside className={`space-y-2 lg:sticky lg:top-[4.2rem] lg:h-[calc(100vh-5rem)] pr-1 lg:pr-2 lg:border-r lg:border-border-subtle/70 ${sidebarCollapsed ? 'sidebar-shell-collapsed' : ''}`}>
            {!sidebarCollapsed && (
            <div className="hidden lg:flex items-center justify-between px-1">
              {!sidebarCollapsed && <p className="panel-title">Data Studio</p>}
              <button
                type="button"
                onClick={() => setSidebarCollapsed((value) => !value)}
                className="sidebar-toggle"
                aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                title={sidebarCollapsed ? 'Expand controls' : 'Collapse controls'}
              >
                {sidebarCollapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
              </button>
            </div>
            )}

            {sidebarCollapsed ? (
              <div className="sidebar-rail hidden lg:flex">
                <button
                  type="button"
                  onClick={() => setSidebarCollapsed(false)}
                  className="sidebar-rail-action"
                  title="Expand controls"
                >
                  <PanelLeftOpen className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={regenerateDataset}
                  className="sidebar-rail-action"
                  title="Resample dataset"
                >
                  <Shuffle className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (taskMode === 'classification') {
                      setShowClassificationDiagnostics(!showClassificationDiagnostics);
                      return;
                    }
                    setShowAssumptions(!showAssumptions);
                  }}
                  className={`sidebar-rail-action ${
                    taskMode === 'classification'
                      ? (showClassificationDiagnostics ? 'sidebar-rail-action-active' : '')
                      : (showAssumptions ? 'sidebar-rail-action-active' : '')
                  }`}
                  title={taskMode === 'classification' ? 'Toggle classification diagnostics' : 'Toggle diagnostics'}
                >
                  <TestTube2 className="w-4 h-4" />
                </button>
                {taskMode === 'regression' && (
                  <button
                    type="button"
                    onClick={() => setCompareWithOls(!compareWithOls)}
                    className={`sidebar-rail-action ${compareWithOls ? 'sidebar-rail-action-active' : ''}`}
                    title="Compare vs OLS"
                  >
                    <Sparkles className="w-4 h-4" />
                  </button>
                )}
              </div>
            ) : (
              <div className="hidden lg:block lg:h-[calc(100%-2rem)] lg:overflow-y-auto">
                <DataControlPanel />
              </div>
            )}

            <div className="lg:hidden">
              <DataControlPanel />
            </div>
          </aside>

          {/* Center Pane: Display */}
          <div className="space-y-3">
            <section className="material-panel p-3 motion-stagger premium-stage" style={{ ['--stagger-index' as any]: 0 }}>
              <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
                <div>
                  <h2 className="text-base font-semibold text-text-primary flex items-center gap-2">
                    <PlayCircle className="w-4 h-4 text-accent" />
                    Live Playground
                  </h2>
                  <p className="text-xs leading-snug text-text-secondary">
                    Touch controls and watch fit behavior change instantly.
                  </p>
                </div>
                <div className="flex flex-wrap gap-1.5 text-xs items-center">
                  <span className="status-pill">Model: {modelType.split('_').join(' ')}</span>
                  <span className="status-pill">Dataset: {dataset}</span>
                  <span className="status-pill">Features: {featureMode === '2d' ? '2D' : '1D'}</span>
                  <span className="status-pill">
                    Eval: {evaluationMode === 'full' ? 'full-fit' : evaluationMode === 'train_test' ? 'train/test' : 'k-fold cv'}
                  </span>
                  <span className="status-pill">n = {data.length}</span>
                  <span className="status-pill">Seed {randomSeed + datasetVersion}</span>
                  <span className="status-pill">Mode: {taskMode}</span>
                  <div className="theme-toggle-shell">
                    <button
                      type="button"
                      className={`theme-toggle-chip ${heroLayoutMode === 'compact' ? 'theme-toggle-chip-active' : ''}`}
                      onClick={() => setHeroLayoutMode('compact')}
                    >
                      Compact
                    </button>
                    <button
                      type="button"
                      className={`theme-toggle-chip ${heroLayoutMode === 'expanded' ? 'theme-toggle-chip-active' : ''}`}
                      onClick={() => setHeroLayoutMode('expanded')}
                    >
                      Expanded
                    </button>
                  </div>
                  <div className="theme-toggle-shell">
                    <button
                      type="button"
                      className={`theme-toggle-chip ${viewMode === 'focus' ? 'theme-toggle-chip-active' : ''}`}
                      onClick={() => setViewMode('focus')}
                    >
                      Focus
                    </button>
                    <button
                      type="button"
                      className={`theme-toggle-chip ${viewMode === 'deep_dive' ? 'theme-toggle-chip-active' : ''}`}
                      onClick={() => setViewMode('deep_dive')}
                    >
                      Deep Dive
                    </button>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-2">
                {heroLayoutMode === 'expanded' && (
                <div className="space-y-1">
                  <p className="text-sm leading-snug text-text-secondary hero-blurb inline-flex items-start gap-1.5">
                    <span className="text-text-primary font-medium">Model:</span> {modelContentMap[modelType].explanation}
                    {featureFlags.ff_info_microcards && (
                      <InfoPopover
                        label="Model Insight"
                        what={modelContentMap[modelType].explanation}
                        why="Model choice controls flexibility, robustness, and how quickly variance rises."
                        tryNext="Toggle Compare OLS and switch to noisy/outlier datasets to inspect tradeoffs."
                      />
                    )}
                  </p>
                  <p className="text-sm leading-snug text-text-secondary hero-blurb inline-flex items-start gap-1.5">
                    <span className="text-text-primary font-medium">Dataset:</span> {datasetExplanations[dataset]}
                    {featureFlags.ff_info_microcards && (
                      <InfoPopover
                        label="Dataset Insight"
                        what={datasetExplanations[dataset]}
                        why="Data shape determines whether model complexity helps or hurts generalization."
                        tryNext="Resample with the same model and compare metric stability before changing hyperparameters."
                      />
                    )}
                  </p>
                </div>
                )}
                <div className="grid grid-cols-2 gap-1.5">
                  <button type="button" onClick={regenerateDataset} className="quick-action">
                    <Shuffle className="w-3.5 h-3.5" />
                    <span>Resample</span>
                  </button>
                  {taskMode === 'classification' ? (
                    <button
                      type="button"
                      onClick={() => setShowClassificationDiagnostics(!showClassificationDiagnostics)}
                      className={`quick-action ${showClassificationDiagnostics ? 'quick-action-active' : ''}`}
                    >
                      <TestTube2 className="w-3.5 h-3.5" />
                      <span>Diagnostics</span>
                    </button>
                  ) : (
                  <button
                    type="button"
                    onClick={() => setShowAssumptions(!showAssumptions)}
                    className={`quick-action ${showAssumptions ? 'quick-action-active' : ''}`}
                  >
                    <TestTube2 className="w-3.5 h-3.5" />
                    <span>Diagnostics</span>
                  </button>
                  )}
                  {taskMode === 'regression' && (
                  <button
                    type="button"
                    onClick={() => setShowOlsSolution(!showOlsSolution)}
                    className={`quick-action ${showOlsSolution ? 'quick-action-active' : ''}`}
                  >
                    <Layers2 className="w-3.5 h-3.5" />
                    <span>OLS Overlay</span>
                  </button>
                  )}
                  {taskMode === 'regression' && (
                  <button
                    type="button"
                    onClick={() => setCompareWithOls(!compareWithOls)}
                    className={`quick-action ${compareWithOls ? 'quick-action-active' : ''}`}
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Compare OLS</span>
                  </button>
                  )}
                </div>
                <div className="flex justify-start">
                  <button
                    type="button"
                    className={`quick-action ${showEquation ? 'quick-action-active' : ''}`}
                    onClick={() => setShowEquation((value) => !value)}
                  >
                    <Layers2 className="w-3.5 h-3.5" />
                    <span>{showEquation ? 'Hide Equation' : 'Show Equation'}</span>
                  </button>
                </div>
                {showEquation && (
                  <div>
                    <p className="panel-title mb-1">Model Equation</p>
                    <FormulaDisplay latex={currentFormula} />
                  </div>
                )}
              </div>
            </section>

            {/* Visualizer */}
            <section className="material-panel p-3 motion-stagger" style={{ ['--stagger-index' as any]: 1 }}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="panel-title">
                  Visualization
                </h3>
                <div className="text-xs text-text-tertiary flex items-center gap-2">
                  <Compass className="w-3.5 h-3.5" />
                  Explore fit behavior across conditions
                </div>
              </div>
              <Visualizer sidebarCollapsed={sidebarCollapsed} />
              <p className="mt-2 text-xs text-text-secondary">
                {taskMode === 'classification'
                  ? 'Colored regions show predicted classes; marker shape highlights correct vs incorrect decisions at the current threshold.'
                  : 'The fit line/plane shows the model prediction; vertical distance from points to the fit corresponds to residual error. In train/test mode, orange points are held-out samples used only for generalization evaluation.'}
              </p>
            </section>

            <section className="material-panel p-3 motion-stagger" style={{ ['--stagger-index' as any]: 2 }}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="panel-title">Metrics</h3>
                <span className="text-xs text-text-tertiary flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  Instant feedback
                </span>
              </div>
              <MetricsBoard compact={viewMode === 'focus'} />
            </section>

            {viewMode === 'focus' && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-1.5">
                  <p className="panel-title">Learning Tools</p>
                  <span className="text-[11px] text-text-tertiary">Open one panel at a time</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <button type="button" onClick={() => setFocusPanel((p) => (p === 'lens' ? 'none' : 'lens'))} className={`quick-action ${focusPanel === 'lens' ? 'quick-action-active' : ''}`}>
                    <Route className="w-3.5 h-3.5" />
                    <span>Lens</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (taskMode === 'classification') setShowClassificationDiagnostics(true);
                      if (taskMode === 'regression') setShowAssumptions(true);
                      setFocusPanel((p) => (p === 'diagnostics' ? 'none' : 'diagnostics'));
                    }}
                    className={`quick-action ${focusPanel === 'diagnostics' ? 'quick-action-active' : ''}`}
                  >
                    <Target className="w-3.5 h-3.5" />
                    <span>Diagnostics</span>
                  </button>
                  <button type="button" onClick={() => setFocusPanel((p) => (p === 'playbook' ? 'none' : 'playbook'))} className={`quick-action ${focusPanel === 'playbook' ? 'quick-action-active' : ''}`}>
                    <BookOpen className="w-3.5 h-3.5" />
                    <span>Playbook</span>
                  </button>
                  {featureFlags.ff_learning_missions && (
                    <button type="button" onClick={() => setFocusPanel((p) => (p === 'missions' ? 'none' : 'missions'))} className={`quick-action ${focusPanel === 'missions' ? 'quick-action-active' : ''}`}>
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>Missions</span>
                    </button>
                  )}
                </div>
              </section>
            )}

            {viewMode === 'focus' && focusPanel === 'lens' && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="panel-title">Bias-Variance Lens</h3>
                </div>
                <LearningInsights sidebarCollapsed={sidebarCollapsed} compact />
              </section>
            )}
            {viewMode === 'focus' && focusPanel === 'diagnostics' && renderDiagnosticsSection(true, true)}
            {viewMode === 'focus' && focusPanel === 'playbook' && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <ClassicalContentPanel compact />
              </section>
            )}
            {viewMode === 'focus' && focusPanel === 'missions' && featureFlags.ff_learning_missions && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <ScenarioMissions compact />
              </section>
            )}

            {viewMode === 'deep_dive' && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="panel-title">
                    Bias-Variance Lens
                  </h3>
                  <div className="text-xs text-text-tertiary">
                    {taskMode === 'classification' ? 'Decision boundaries and threshold effects' : 'Underfit vs overfit behavior'}
                  </div>
                </div>
                <LearningInsights sidebarCollapsed={sidebarCollapsed} compact />
              </section>
            )}

            {viewMode === 'deep_dive' && renderDiagnosticsSection(true, true)}

            {viewMode === 'deep_dive' && featureFlags.ff_learning_missions && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <ScenarioMissions />
              </section>
            )}

            {viewMode === 'deep_dive' && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="panel-title">Classical ML Playbook</h3>
                  <div className="text-xs text-text-tertiary">Action-focused learning cues</div>
                </div>
                <ClassicalContentPanel />
              </section>
            )}

            {viewMode === 'deep_dive' && (
              <div className="grid grid-cols-1 gap-3">
                <section className="material-panel p-3 motion-stagger" style={{ ['--stagger-index' as any]: 5 }}>
                  <h3 className="panel-title mb-2">
                    Python Export
                  </h3>
                  <CodeExporter />
                </section>
              </div>
            )}
          </div>

          {/* Right Pane: Model Sidebar */}
          <aside className="space-y-2 lg:sticky lg:top-[4.2rem] lg:h-[calc(100vh-5rem)] pl-1">
            <div className="hidden lg:flex items-center justify-between px-1">
              <p className="panel-title">Model Studio</p>
            </div>
            <div className="hidden lg:block lg:h-[calc(100%-2rem)] lg:overflow-y-auto">
              <ModelControlPanel />
            </div>
            <div className="lg:hidden">
              <ModelControlPanel />
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}
