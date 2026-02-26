import { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { useModelStore } from '../store/modelStore';
import { generateDataset, evaluateModelMetrics, fitRegressionModel, latexForModel, supports2D } from '../lib/dataUtils';
import { DataControlPanel, ModelControlPanel } from '../components/ControlPanel/ControlPanel';
import { FormulaDisplay } from '../components/FormulaDisplay';
import { Visualizer } from '../components/Visualizer';
import { MetricsBoard } from '../components/MetricsBoard';
import { InfoPopover } from '../components/InfoPopover';
import { Toaster } from '../components/ui/sonner';
import { toast } from 'sonner';
import { Compass, Sparkles, PlayCircle, TestTube2, Layers2, Shuffle, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, BookOpen, Route, Target, RotateCcw } from 'lucide-react';
import { ThemeToggle } from '../components/ThemeToggle';
import { datasetExplanations, modelContentMap } from '../content/classicalContentAdapter';
import { featureFlags } from '../config/featureFlags';

const LazyCodeExporter = lazy(() => import('../components/CodeExporter').then((m) => ({ default: m.CodeExporter })));
const LazyAssumptionChecker = lazy(() => import('../components/AssumptionChecker/AssumptionChecker').then((m) => ({ default: m.AssumptionChecker })));
const LazyClassificationDiagnostics = lazy(() => import('../components/ClassificationDiagnostics').then((m) => ({ default: m.ClassificationDiagnostics })));
const LazyLearningInsights = lazy(() => import('../components/LearningInsights').then((m) => ({ default: m.LearningInsights })));
const LazyClassicalContentPanel = lazy(() => import('../components/ClassicalContentPanel').then((m) => ({ default: m.ClassicalContentPanel })));
const LazyScenarioMissions = lazy(() => import('../components/ScenarioMissions').then((m) => ({ default: m.ScenarioMissions })));
const LazyDimensionalityReductionLab = lazy(() => import('../components/DimensionalityReductionLab').then((m) => ({ default: m.DimensionalityReductionLab })));

export function InteractiveModelTemplate() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(false);
  const [focusPanel, setFocusPanel] = useState<'none' | 'lens' | 'diagnostics' | 'playbook' | 'missions'>('none');
  const [showEquation, setShowEquation] = useState(false);
  const [tourStep, setTourStep] = useState(0);
  const [showLoadGuardHint, setShowLoadGuardHint] = useState(false);
  const [renderLensSection, setRenderLensSection] = useState(false);
  const [renderDiagnosticsPanel, setRenderDiagnosticsPanel] = useState(false);
  const [renderMissionsSection, setRenderMissionsSection] = useState(false);
  const [renderPlaybookSection, setRenderPlaybookSection] = useState(false);
  const [renderExportSection, setRenderExportSection] = useState(false);
  const [renderDimReductionSection, setRenderDimReductionSection] = useState(false);
  const [tourRect, setTourRect] = useState<DOMRect | null>(null);
  const {
    taskMode,
    setTaskMode,
    modelType,
    setModelType,
    dataset,
    params,
    setParam,
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
    comparePinnedModels,
    togglePinnedModel,
    heroLayoutMode,
    setHeroLayoutMode,
    viewMode,
    onboardingState,
    setOnboardingState,
    randomDataRecipe,
    setData,
    setMetrics,
    error,
    setError,
    setShowAssumptions,
    setShowClassificationDiagnostics,
    setShowOlsSolution,
    setCompareWithOls,
    resetParams,
    regenerateDataset,
  } = useModelStore();

  // Generate data when dataset changes
  useEffect(() => {
    try {
      const newData = generateDataset(dataset, sampleSize, randomSeed + datasetVersion, featureMode, randomDataRecipe);
      setData(newData);
      setError(null);
    } catch {
      setError('Failed to generate dataset. Please try again.');
    }
  }, [dataset, sampleSize, randomSeed, datasetVersion, featureMode, randomDataRecipe, setData, setError]);

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
    if (viewMode === 'deep_dive') setFocusPanel('none');
  }, [viewMode]);

  useEffect(() => {
    const handleGuard = () => {
      const isSmall = window.innerWidth < 1200;
      if (!isSmall) {
        setShowLoadGuardHint(false);
        return;
      }
      const openPanelCount = (showEquation ? 1 : 0)
        + (showAssumptions ? 1 : 0)
        + (showClassificationDiagnostics ? 1 : 0)
        + (showOlsSolution ? 1 : 0)
        + (compareWithOls ? 1 : 0)
        + (viewMode === 'deep_dive' ? 2 : 0);
      setShowLoadGuardHint(openPanelCount >= 4 && viewMode === 'deep_dive');
    };
    handleGuard();
    window.addEventListener('resize', handleGuard);
    return () => window.removeEventListener('resize', handleGuard);
  }, [
    viewMode,
    showEquation,
    showAssumptions,
    showClassificationDiagnostics,
    showOlsSolution,
    compareWithOls,
  ]);

  const fittedForFormula = useMemo(
    () => (data.length > 0 ? fitRegressionModel(data, modelType, params) : null),
    [data, modelType, params]
  );
  const currentFormula = latexForModel(modelType, params, fittedForFormula);
  const dimensionalityCompare = useMemo(() => {
    if (taskMode !== 'regression' || data.length === 0) return null;
    const pcrMetrics = evaluateModelMetrics(
      data,
      'pcr_regressor',
      { ...params, pcaComponents: params.pcaComponents ?? 2 },
      evaluationMode,
      testRatio,
      cvFolds,
      randomSeed + datasetVersion
    );
    const plsMetrics = evaluateModelMetrics(
      data,
      'pls_regressor',
      { ...params, plsComponents: params.plsComponents ?? 2 },
      evaluationMode,
      testRatio,
      cvFolds,
      randomSeed + datasetVersion
    );
    return { pcrMetrics, plsMetrics };
  }, [taskMode, data, params, evaluationMode, testRatio, cvFolds, randomSeed, datasetVersion]);
  const fitSuitability = useMemo(() => {
    if (taskMode !== 'regression') return 'Good match';
    if (featureMode === '2d' && !supports2D(modelType)) return 'Likely underfit';
    if ((dataset === 'quadratic' || dataset === 'sinusoidal') && ['ols', 'ridge', 'lasso', 'elasticnet'].includes(modelType)) return 'Likely underfit';
    if ((dataset === 'noisy' || dataset === 'outliers') && modelType === 'ols') return 'Likely high variance';
    return 'Good match';
  }, [taskMode, featureMode, modelType, dataset]);
  const tourSteps = [
    { selector: "[data-tour='mode-switch']", title: 'Task Mode', text: 'Switch between regression and classification workflows here.' },
    { selector: "[data-tour='theme-toggle']", title: 'Theme & Tour', text: 'Use light/dark theme and replay this guided tour anytime.' },
    { selector: "[data-tour='data-studio']", title: 'Data Studio', text: 'Start here: choose dataset, feature space, and sampling settings.' },
    { selector: "[data-tour='model-studio']", title: 'Model Studio', text: 'Choose model family, model, and hyperparameters here.' },
    { selector: "[data-tour='hero-actions']", title: 'Live Controls', text: 'Quick actions for resampling, OLS overlay, and equation view.' },
    { selector: "[data-tour='visualization']", title: 'Visualization', text: 'Watch boundaries and fit behavior update from your changes.' },
    { selector: "[data-tour='metrics']", title: 'Metrics', text: 'Confirm impact with objective metrics and the delta caption.' },
    { selector: "[data-tour='lens']", title: 'Bias-Variance Lens', text: 'Inspect complexity tradeoffs and error patterns.' },
    { selector: "[data-tour='diagnostics']", title: 'Diagnostics', text: 'Use confusion/residual diagnostics to validate model behavior.' },
    { selector: "[data-tour='dim-lab']", title: 'Dimensionality Lab', text: 'Compare PCR/PLS and inspect PCA variance/scatter views.' },
    { selector: "[data-tour='missions']", title: 'Learning Missions', text: 'Apply challenge presets with completion criteria.' },
    { selector: "[data-tour='playbook']", title: 'Playbook', text: 'Use guided “try next” actions tied to your model context.' },
    { selector: "[data-tour='export']", title: 'Python Export', text: 'Export the current sandbox configuration to Python code.' },
  ] as const;

  const findVisibleTourTarget = (selector: string): Element | null => {
    const nodes = Array.from(document.querySelectorAll(selector));
    return nodes.find((node) => {
      const rect = node.getBoundingClientRect();
      return rect.width > 8 && rect.height > 8;
    }) ?? null;
  };

  useEffect(() => {
    const seen = window.localStorage.getItem('mls_tour_seen');
    if (!seen) {
      setOnboardingState('in_progress');
      setTourStep(0);
    }
  }, [setOnboardingState]);

  useEffect(() => {
    if (onboardingState !== 'in_progress') {
      setTourRect(null);
      return;
    }
    setRenderLensSection(true);
    setRenderDiagnosticsPanel(true);
    setRenderDimReductionSection(true);
    setRenderMissionsSection(true);
    setRenderPlaybookSection(true);
    setRenderExportSection(true);
    const step = tourSteps[Math.max(0, Math.min(tourStep, tourSteps.length - 1))];
    const updateRect = () => {
      const visibleNode = findVisibleTourTarget(step.selector);
      if (!visibleNode) {
        setTourRect(null);
        return;
      }
      setTourRect(visibleNode.getBoundingClientRect());
    };
    updateRect();
    window.addEventListener('resize', updateRect);
    window.addEventListener('scroll', updateRect, true);
    const raf = window.requestAnimationFrame(updateRect);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener('resize', updateRect);
      window.removeEventListener('scroll', updateRect, true);
    };
  }, [onboardingState, tourStep]);

  useEffect(() => {
    if (onboardingState !== 'in_progress') return;
    const step = tourSteps[Math.max(0, Math.min(tourStep, tourSteps.length - 1))];
    const target = findVisibleTourTarget(step.selector);
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
  }, [onboardingState, tourStep]);

  useEffect(() => {
    if (viewMode !== 'deep_dive') return;
    setRenderLensSection(true);
    const t1 = window.setTimeout(() => setRenderDiagnosticsPanel(true), 80);
    const t2 = window.setTimeout(() => setRenderDimReductionSection(true), 140);
    const t3 = window.setTimeout(() => setRenderMissionsSection(true), 200);
    const t4 = window.setTimeout(() => setRenderPlaybookSection(true), 260);
    const t5 = window.setTimeout(() => setRenderExportSection(true), 320);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
      window.clearTimeout(t4);
      window.clearTimeout(t5);
    };
  }, [viewMode]);

  const lazyFallback = <div className="rounded-xl border border-border-subtle p-2.5 text-xs text-text-tertiary">Loading panel...</div>;

  const renderDiagnosticsSection = (compact = false, forceVisible = false) => {
    if (taskMode === 'regression' && (showAssumptions || forceVisible)) {
      return (
        <section className={`material-panel ${compact ? 'p-2.5' : 'p-3'} motion-stagger`} style={{ ['--stagger-index' as any]: 2 }}>
          <Suspense fallback={lazyFallback}>
            <LazyAssumptionChecker sidebarCollapsed={sidebarCollapsed} compact={compact} />
          </Suspense>
        </section>
      );
    }
    if (taskMode === 'classification' && (showClassificationDiagnostics || forceVisible)) {
      return (
        <section className={`material-panel ${compact ? 'p-2.5' : 'p-3'} motion-stagger`} style={{ ['--stagger-index' as any]: 2 }}>
          <Suspense fallback={lazyFallback}>
            <LazyClassificationDiagnostics sidebarCollapsed={sidebarCollapsed} compact={compact} />
          </Suspense>
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
              <div data-tour="mode-switch" className="theme-toggle-shell">
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
              <button
                type="button"
                className="theme-toggle-chip"
                onClick={() => {
                  setOnboardingState('in_progress');
                  setTourStep(0);
                }}
              >
                Tour
              </button>
              <div data-tour="theme-toggle">
                <ThemeToggle />
              </div>
            </nav>
          </div>
          {onboardingState === 'in_progress' && (
            <section className="mt-2 material-panel-soft px-3 py-2 text-xs text-text-secondary">
              Guided tour is active. Follow highlighted bubbles.
            </section>
          )}
          {showLoadGuardHint && (
            <section className="mt-2 material-panel-soft px-3 py-2 text-xs text-text-secondary flex items-center justify-between gap-2">
              <span>Cognitive Load Guard suggestion: many panels are open for this viewport. Consider collapsing extras for readability.</span>
              <button type="button" className="quick-action" onClick={() => setShowLoadGuardHint(false)}>
                Dismiss
              </button>
            </section>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="w-full px-3 lg:pl-0 lg:pr-4 py-3">
        <div className={`grid grid-cols-1 gap-4 lg:transition-[grid-template-columns] lg:duration-300 lg:ease-out ${
          sidebarCollapsed && rightSidebarCollapsed
            ? 'lg:grid-cols-[64px_minmax(0,1fr)_64px]'
            : sidebarCollapsed
              ? 'lg:grid-cols-[64px_minmax(0,1fr)_minmax(250px,18%)]'
              : rightSidebarCollapsed
                ? 'lg:grid-cols-[minmax(250px,18%)_minmax(0,1fr)_64px]'
                : 'lg:grid-cols-[minmax(250px,18%)_minmax(0,1fr)_minmax(250px,18%)]'
        }`}>
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
              <div data-tour="data-studio" className="hidden lg:block lg:h-[calc(100%-2rem)] lg:overflow-y-auto">
                <DataControlPanel />
              </div>
            )}

            <div data-tour="data-studio" className="lg:hidden">
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
                  <button
                    type="button"
                    onClick={() => togglePinnedModel(modelType)}
                    className={`status-pill ${comparePinnedModels.includes(modelType) ? 'border-accent/50 text-accent' : ''}`}
                  >
                    {comparePinnedModels.includes(modelType) ? 'Pinned' : 'Pin model'}
                  </button>
                  {taskMode === 'regression' && <span className="status-pill">Fit: {fitSuitability}</span>}
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
                <div data-tour="hero-actions" className={`grid gap-1.5 ${taskMode === 'regression' ? 'grid-cols-2' : 'grid-cols-1'}`}>
                  <button type="button" onClick={regenerateDataset} className="quick-action">
                    <Shuffle className="w-3.5 h-3.5" />
                    <span>Resample</span>
                  </button>
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
            <section data-tour="visualization" className="material-panel p-3 motion-stagger" style={{ ['--stagger-index' as any]: 1 }}>
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

            <section data-tour="metrics" className="material-panel p-3 motion-stagger" style={{ ['--stagger-index' as any]: 2 }}>
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
                <Suspense fallback={lazyFallback}>
                  <LazyLearningInsights sidebarCollapsed={sidebarCollapsed} compact />
                </Suspense>
              </section>
            )}
            {viewMode === 'focus' && focusPanel === 'diagnostics' && renderDiagnosticsSection(true, true)}
            {viewMode === 'focus' && focusPanel === 'playbook' && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <Suspense fallback={lazyFallback}>
                  <LazyClassicalContentPanel compact />
                </Suspense>
              </section>
            )}
            {viewMode === 'focus' && focusPanel === 'missions' && featureFlags.ff_learning_missions && (
              <section className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <Suspense fallback={lazyFallback}>
                  <LazyScenarioMissions compact />
                </Suspense>
              </section>
            )}

            {viewMode === 'deep_dive' && renderLensSection && (
              <section data-tour="lens" className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="panel-title">
                    Bias-Variance Lens
                  </h3>
                  <div className="text-xs text-text-tertiary">
                    {taskMode === 'classification' ? 'Decision boundaries and threshold effects' : 'Underfit vs overfit behavior'}
                  </div>
                </div>
                <Suspense fallback={lazyFallback}>
                  <LazyLearningInsights sidebarCollapsed={sidebarCollapsed} compact />
                </Suspense>
              </section>
            )}

            {viewMode === 'deep_dive' && renderDiagnosticsPanel && (
              <div data-tour="diagnostics">
                {renderDiagnosticsSection(true, true)}
              </div>
            )}

            {viewMode === 'deep_dive' && renderDimReductionSection && taskMode === 'regression' && dimensionalityCompare && (
              <section data-tour="dim-lab" className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="panel-title">Dimensionality Reduction Lab</h3>
                  <span className="text-[11px] text-text-tertiary">PCR and PLS on current data split</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  <button
                    type="button"
                    className={`mission-card text-left ${modelType === 'pcr_regressor' ? 'mission-card-success' : ''}`}
                    onClick={() => {
                      setModelType('pcr_regressor');
                      setParam('pcaComponents', Math.max(1, params.pcaComponents ?? 2));
                    }}
                  >
                    <p className="text-sm font-medium text-text-primary">PCR</p>
                    <p className="text-xs text-text-secondary mt-1">R² {dimensionalityCompare.pcrMetrics.r2.toFixed(3)} · RMSE {dimensionalityCompare.pcrMetrics.rmse.toFixed(3)}</p>
                    <p className="text-[11px] text-text-tertiary mt-1">Principal components then linear regression.</p>
                  </button>
                  <button
                    type="button"
                    className={`mission-card text-left ${modelType === 'pls_regressor' ? 'mission-card-success' : ''}`}
                    onClick={() => {
                      setModelType('pls_regressor');
                      setParam('plsComponents', Math.max(1, params.plsComponents ?? 2));
                    }}
                  >
                    <p className="text-sm font-medium text-text-primary">PLS</p>
                    <p className="text-xs text-text-secondary mt-1">R² {dimensionalityCompare.plsMetrics.r2.toFixed(3)} · RMSE {dimensionalityCompare.plsMetrics.rmse.toFixed(3)}</p>
                    <p className="text-[11px] text-text-tertiary mt-1">Latent factors aligned with target variation.</p>
                  </button>
                </div>
                <div className="mt-2">
                  <Suspense fallback={lazyFallback}>
                    <LazyDimensionalityReductionLab />
                  </Suspense>
                </div>
              </section>
            )}

            {viewMode === 'deep_dive' && renderMissionsSection && featureFlags.ff_learning_missions && (
              <section data-tour="missions" className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <Suspense fallback={lazyFallback}>
                  <LazyScenarioMissions />
                </Suspense>
              </section>
            )}

            {viewMode === 'deep_dive' && renderPlaybookSection && (
              <section data-tour="playbook" className="material-panel p-2.5 motion-stagger" style={{ ['--stagger-index' as any]: 3 }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="panel-title">Classical ML Playbook</h3>
                  <div className="text-xs text-text-tertiary">Action-focused learning cues</div>
                </div>
                <Suspense fallback={lazyFallback}>
                  <LazyClassicalContentPanel />
                </Suspense>
              </section>
            )}

            {viewMode === 'deep_dive' && renderExportSection && (
              <div className="grid grid-cols-1 gap-3">
                <section data-tour="export" className="material-panel p-3 motion-stagger" style={{ ['--stagger-index' as any]: 5 }}>
                  <h3 className="panel-title mb-2">
                    Python Export
                  </h3>
                  <Suspense fallback={lazyFallback}>
                    <LazyCodeExporter />
                  </Suspense>
                </section>
              </div>
            )}
          </div>

          {/* Right Pane: Model Sidebar */}
          <aside className={`space-y-2 lg:sticky lg:top-[4.2rem] lg:h-[calc(100vh-5rem)] pl-1 lg:pl-2 lg:border-l lg:border-border-subtle/70 ${rightSidebarCollapsed ? 'sidebar-shell-collapsed' : ''}`}>
            {!rightSidebarCollapsed && (
              <div className="hidden lg:flex items-center justify-between px-1">
                <p className="panel-title">Model Studio</p>
                <button
                  type="button"
                  onClick={() => setRightSidebarCollapsed((value) => !value)}
                  className="sidebar-toggle"
                  aria-label={rightSidebarCollapsed ? 'Expand model sidebar' : 'Collapse model sidebar'}
                  title={rightSidebarCollapsed ? 'Expand model controls' : 'Collapse model controls'}
                >
                  {rightSidebarCollapsed ? <PanelRightOpen className="w-4 h-4" /> : <PanelRightClose className="w-4 h-4" />}
                </button>
              </div>
            )}
            {rightSidebarCollapsed ? (
              <div className="sidebar-rail hidden lg:flex">
                <button
                  type="button"
                  onClick={() => setRightSidebarCollapsed(false)}
                  className="sidebar-rail-action"
                  title="Expand model controls"
                >
                  <PanelRightOpen className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={resetParams}
                  className="sidebar-rail-action"
                  title="Reset model params"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                {taskMode === 'regression' && (
                  <button
                    type="button"
                    onClick={() => setShowOlsSolution(!showOlsSolution)}
                    className={`sidebar-rail-action ${showOlsSolution ? 'sidebar-rail-action-active' : ''}`}
                    title="Toggle OLS overlay"
                  >
                    <Layers2 className="w-4 h-4" />
                  </button>
                )}
              </div>
            ) : (
              <div data-tour="model-studio" className="hidden lg:block lg:h-[calc(100%-2rem)] lg:overflow-y-auto">
                <ModelControlPanel />
              </div>
            )}
            <div data-tour="model-studio" className="lg:hidden">
              <ModelControlPanel />
            </div>
          </aside>
        </div>
      </main>
      {onboardingState === 'in_progress' && (
        <div className="fixed inset-0 z-[100] pointer-events-none">
          <div className="absolute inset-0 bg-black/25" />
          {tourRect && (
            <div
              className="absolute rounded-xl border-2 border-accent shadow-[0_0_0_9999px_rgba(0,0,0,0.18)]"
              style={{
                left: Math.max(8, tourRect.left - 4),
                top: Math.max(8, tourRect.top - 4),
                width: Math.max(20, tourRect.width + 8),
                height: Math.max(20, tourRect.height + 8),
              }}
            />
          )}
          <div
            className="absolute pointer-events-auto w-[min(92vw,320px)] material-panel p-2.5"
            style={{
              left: tourRect ? Math.max(8, Math.min(window.innerWidth - 328, tourRect.left)) : 12,
              top: tourRect ? Math.max(8, Math.min(window.innerHeight - 180, tourRect.bottom + 12)) : 72,
            }}
          >
            <div className="flex items-center justify-between mb-1">
              <p className="panel-title">Guided Tour</p>
              <span className="text-[11px] text-text-tertiary">{tourStep + 1}/{tourSteps.length}</span>
            </div>
            <p className="text-sm font-medium text-text-primary">{tourSteps[tourStep].title}</p>
            <p className="text-xs text-text-secondary mt-1">{tourSteps[tourStep].text}</p>
            <div className="flex gap-1.5 mt-2">
              <button type="button" className="quick-action" onClick={() => setTourStep((step) => Math.max(0, step - 1))}>Back</button>
              <button
                type="button"
                className="quick-action quick-action-active"
                onClick={() => {
                  if (tourStep >= tourSteps.length - 1) {
                    setOnboardingState('completed');
                    window.localStorage.setItem('mls_tour_seen', '1');
                    return;
                  }
                  setTourStep((step) => step + 1);
                }}
              >
                {tourStep >= tourSteps.length - 1 ? 'Finish' : 'Next'}
              </button>
              <button
                type="button"
                className="quick-action"
                onClick={() => {
                  setOnboardingState('completed');
                  window.localStorage.setItem('mls_tour_seen', '1');
                }}
              >
                Skip
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
