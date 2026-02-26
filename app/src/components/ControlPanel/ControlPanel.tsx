import { useState, type ReactNode } from 'react';
import { useModelStore, type DatasetType, type MetricKey, type ModelType } from '../../store/modelStore';
import { Slider } from './Slider';
import { Toggle } from './Toggle';
import {
  AlertTriangle,
  BarChart3,
  Database,
  FlaskConical,
  Filter,
  RefreshCw,
  RotateCcw,
  Settings2,
  Sliders,
  ChevronDown,
} from 'lucide-react';
import { metricMeta, modelContentMap } from '../../content/classicalContentAdapter';
import { InfoPopover } from '../InfoPopover';
import { featureFlags } from '../../config/featureFlags';

const regressionModelOptions: { value: ModelType; label: string; description: string; family: 'linear' | 'regularized' | 'nonlinear' }[] = [
  { value: 'ols', label: modelContentMap.ols.title, description: modelContentMap.ols.explanation, family: 'linear' },
  { value: 'ridge', label: modelContentMap.ridge.title, description: modelContentMap.ridge.explanation, family: 'regularized' },
  { value: 'lasso', label: modelContentMap.lasso.title, description: modelContentMap.lasso.explanation, family: 'regularized' },
  { value: 'elasticnet', label: modelContentMap.elasticnet.title, description: modelContentMap.elasticnet.explanation, family: 'regularized' },
  { value: 'polynomial', label: modelContentMap.polynomial.title, description: modelContentMap.polynomial.explanation, family: 'nonlinear' },
  { value: 'forward_stepwise', label: modelContentMap.forward_stepwise.title, description: modelContentMap.forward_stepwise.explanation, family: 'nonlinear' },
  { value: 'backward_stepwise', label: modelContentMap.backward_stepwise.title, description: modelContentMap.backward_stepwise.explanation, family: 'nonlinear' },
  { value: 'svm_regressor', label: modelContentMap.svm_regressor.title, description: modelContentMap.svm_regressor.explanation, family: 'regularized' },
  { value: 'pcr_regressor', label: modelContentMap.pcr_regressor.title, description: modelContentMap.pcr_regressor.explanation, family: 'regularized' },
  { value: 'pls_regressor', label: modelContentMap.pls_regressor.title, description: modelContentMap.pls_regressor.explanation, family: 'regularized' },
];

const classificationModelOptions: { value: ModelType; label: string; description: string; family: 'linear' | 'local' | 'margin' | 'tree' | 'boosting' }[] = [
  { value: 'logistic_classifier', label: modelContentMap.logistic_classifier.title, description: modelContentMap.logistic_classifier.explanation, family: 'linear' },
  { value: 'knn_classifier', label: modelContentMap.knn_classifier.title, description: modelContentMap.knn_classifier.explanation, family: 'local' },
  { value: 'svm_classifier', label: modelContentMap.svm_classifier.title, description: modelContentMap.svm_classifier.explanation, family: 'margin' },
  { value: 'decision_tree_classifier', label: modelContentMap.decision_tree_classifier.title, description: modelContentMap.decision_tree_classifier.explanation, family: 'tree' },
  { value: 'random_forest_classifier', label: modelContentMap.random_forest_classifier.title, description: modelContentMap.random_forest_classifier.explanation, family: 'tree' },
  { value: 'adaboost_classifier', label: modelContentMap.adaboost_classifier.title, description: modelContentMap.adaboost_classifier.explanation, family: 'boosting' },
  { value: 'gradient_boosting_classifier', label: modelContentMap.gradient_boosting_classifier.title, description: modelContentMap.gradient_boosting_classifier.explanation, family: 'boosting' },
];

const regressionDatasetOptions: { value: DatasetType; label: string; description: string }[] = [
  { value: 'linear', label: 'Linear', description: 'Mostly linear signal' },
  { value: 'noisy', label: 'Noisy', description: 'High random variance' },
  { value: 'outliers', label: 'Outliers', description: 'Heavy point anomalies' },
  { value: 'heteroscedastic', label: 'Heteroscedastic', description: 'Variance grows with x' },
  { value: 'quadratic', label: 'Quadratic', description: 'Curved relationship' },
  { value: 'sinusoidal', label: 'Sinusoidal', description: 'Periodic trend + noise' },
  { value: 'piecewise', label: 'Piecewise', description: 'Segmented linear regimes' },
  { value: 'random_recipe', label: 'Random Recipe', description: 'User-generated synthetic pattern/noise mix' },
];

const classificationDatasetOptions: { value: DatasetType; label: string; description: string }[] = [
  { value: 'class_linear', label: 'Linear Classes', description: 'Easy linear separation' },
  { value: 'class_overlap', label: 'Overlap', description: 'Boundary ambiguity' },
  { value: 'class_moons', label: 'Moons', description: 'Nonlinear structure' },
  { value: 'class_imbalanced', label: 'Imbalanced', description: 'Rare positive class' },
];

type RegressionFamilyFilter = 'linear' | 'regularized' | 'nonlinear';
type ClassificationFamilyFilter = 'linear' | 'local' | 'margin' | 'tree' | 'boosting';
const regressionMetricOptions: Array<{ key: MetricKey; label: string }> = [
  { key: 'r2', label: metricMeta.r2.label },
  { key: 'adjustedR2', label: metricMeta.adjustedR2.label },
  { key: 'rmse', label: metricMeta.rmse.label },
  { key: 'mae', label: metricMeta.mae.label },
  { key: 'mse', label: metricMeta.mse.label },
  { key: 'mape', label: metricMeta.mape.label },
  { key: 'medianAe', label: metricMeta.medianAe.label },
  { key: 'explainedVariance', label: 'Expl.Var' },
];
const classificationMetricOptions: Array<{ key: MetricKey; label: string }> = [
  { key: 'accuracy', label: metricMeta.accuracy.label },
  { key: 'precision', label: metricMeta.precision.label },
  { key: 'recall', label: metricMeta.recall.label },
  { key: 'specificity', label: metricMeta.specificity.label },
  { key: 'f1', label: metricMeta.f1.label },
  { key: 'rocAuc', label: metricMeta.rocAuc.label },
  { key: 'prAuc', label: metricMeta.prAuc.label },
  { key: 'logLoss', label: metricMeta.logLoss.label },
];

function useControlStore() {
  const store = useModelStore();
  const usesRegularization = store.modelType === 'ridge' || store.modelType === 'lasso' || store.modelType === 'elasticnet';
  const usesL1Ratio = store.modelType === 'elasticnet';
  const usesPolynomialDegree =
    store.modelType === 'polynomial' || store.modelType === 'forward_stepwise' || store.modelType === 'backward_stepwise';
  const usesStepwiseTerms = store.modelType === 'forward_stepwise' || store.modelType === 'backward_stepwise';

  return {
    ...store,
    usesRegularization,
    usesL1Ratio,
    usesPolynomialDegree,
    usesStepwiseTerms,
  };
}

function CollapsibleTile({
  title,
  icon,
  defaultOpen = false,
  children,
  headerRight,
}: {
  title: string;
  icon: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  headerRight?: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="material-panel p-3.5">
      <button type="button" onClick={() => setOpen((value) => !value)} className="w-full flex items-center justify-between gap-2 text-left">
        <span className="inline-flex items-center gap-2">
          {icon}
          <span className="panel-title">{title}</span>
        </span>
        <span className="inline-flex items-center gap-2">
          {headerRight}
          <ChevronDown className={`w-3.5 h-3.5 text-text-tertiary transition-transform ${open ? 'rotate-180' : ''}`} />
        </span>
      </button>
      {open && <div className="mt-3">{children}</div>}
    </section>
  );
}

export function DataControlPanel() {
  const {
    taskMode,
    dataset,
    setDataset,
    sampleSize,
    setSampleSize,
    randomSeed,
    setRandomSeed,
    evaluationMode,
    setEvaluationMode,
    testRatio,
    setTestRatio,
    cvFolds,
    setCvFolds,
    featureMode,
    setFeatureMode,
    regenerateDataset,
    randomDataRecipe,
    setRandomDataRecipe,
    generateRandomDatasetFromRecipe,
  } = useControlStore();
  const datasetOptions = taskMode === 'classification' ? classificationDatasetOptions : regressionDatasetOptions;
  const selectedDataset = datasetOptions.find((option) => option.value === dataset) ?? datasetOptions[0];

  return (
    <div className="space-y-3">
      <CollapsibleTile
        title="Data Profile"
        icon={<Database className="w-4 h-4 text-accent" />}
        defaultOpen
        headerRight={<span className="text-[11px] text-text-tertiary">{selectedDataset.label}</span>}
      >
        {taskMode === 'regression' ? (
          <div className="grid grid-cols-2 gap-1.5 mb-2.5">
            <button type="button" onClick={() => setFeatureMode('1d')} className={`eval-chip ${featureMode === '1d' ? 'eval-chip-active' : ''}`}>1D Features</button>
            <button
              type="button"
              onClick={() => {
                setFeatureMode('2d');
              }}
              className={`eval-chip ${featureMode === '2d' ? 'eval-chip-active' : ''}`}
            >
              2D Features
            </button>
          </div>
        ) : (
          <div className="mb-2.5">
            <span className="status-pill">Feature Space: 2D Decision View</span>
          </div>
        )}
        {taskMode === 'classification' && (
          <p className="text-[11px] text-text-tertiary mb-2">Classification mode is optimized for 2D boundary visualization.</p>
        )}
        <div className="material-panel-soft p-2 mb-2">
          <p className="text-xs text-text-secondary">{selectedDataset.description}</p>
          <p className="text-[11px] text-text-tertiary mt-1">
            Signal: {dataset === 'linear' ? 'Linear' : dataset === 'quadratic' ? 'Curved' : dataset === 'sinusoidal' ? 'Periodic' : dataset === 'random_recipe' ? 'Custom Recipe' : 'Mixed'} ·
            Noise: {dataset === 'noisy' || dataset === 'random_recipe' ? 'Medium/High' : 'Low/Medium'} ·
            Outliers: {dataset === 'outliers' ? 'High' : dataset === 'random_recipe' ? randomDataRecipe.outlierLevel : 'Low'}
          </p>
        </div>
        <div className="space-y-1.5">
          {datasetOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setDataset(option.value)}
              className={`control-option interactive-lift w-full text-left px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                dataset === option.value
                  ? 'control-option-active'
                  : ''
              }`}
            >
              <div className="text-sm font-medium text-text-primary">{option.label}</div>
              <div className="text-xs leading-snug text-text-secondary mt-0.5">{option.description}</div>
            </button>
          ))}
        </div>
      </CollapsibleTile>

      <CollapsibleTile title="Resampling" icon={<BarChart3 className="w-4 h-4 text-accent" />} defaultOpen>
        <div className="flex items-center gap-2 mb-3">
          <span className="panel-title">Train & Validate</span>
          {featureFlags.ff_info_microcards && (
            <InfoPopover
              label="Resampling Modes"
              what="Choose how performance is estimated: full fit, train/test split, or k-fold cross-validation."
              why="Better resampling gives a more trustworthy view of generalization."
              tryNext="Switch between train/test and k-fold while keeping the same model and dataset."
            />
          )}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 mb-3">
          <button
            type="button"
            onClick={() => setEvaluationMode('full')}
            className={`eval-chip !px-2 !py-1.5 text-[11px] leading-tight whitespace-normal text-center ${evaluationMode === 'full' ? 'eval-chip-active' : ''}`}
          >
            Full Fit
          </button>
          <button
            type="button"
            onClick={() => setEvaluationMode('train_test')}
            className={`eval-chip !px-2 !py-1.5 text-[11px] leading-tight whitespace-normal text-center ${evaluationMode === 'train_test' ? 'eval-chip-active' : ''}`}
          >
            Train/<wbr />Test
          </button>
          <button
            type="button"
            onClick={() => setEvaluationMode('cross_validation')}
            className={`eval-chip col-span-2 sm:col-span-1 !px-2 !py-1.5 text-[11px] leading-tight whitespace-normal text-center ${evaluationMode === 'cross_validation' ? 'eval-chip-active' : ''}`}
          >
            K-Fold<wbr /> CV
          </button>
        </div>
        {evaluationMode === 'train_test' && (
          <Slider
            label="Test Split Ratio"
            value={testRatio}
            min={0.1}
            max={0.45}
            step={0.01}
            onChange={(value) => setTestRatio(Number(value.toFixed(2)))}
          />
        )}
        {evaluationMode === 'cross_validation' && (
          <Slider
            label="CV Folds"
            value={cvFolds}
            min={3}
            max={10}
            step={1}
            onChange={(value) => setCvFolds(Math.round(value))}
          />
        )}
      </CollapsibleTile>

      <CollapsibleTile title="Data Generation" icon={<FlaskConical className="w-4 h-4 text-accent" />}>
        <div className="space-y-3.5">
          <Slider
            label="Sample Size (n)"
            value={sampleSize}
            min={30}
            max={320}
            step={10}
            onChange={(value) => setSampleSize(Math.round(value))}
          />
          <Slider
            label="Random Seed"
            value={randomSeed}
            min={1}
            max={9999}
            step={1}
            onChange={(value) => setRandomSeed(Math.round(value))}
          />
          <button
            type="button"
            onClick={regenerateDataset}
            className="control-option interactive-lift w-full inline-flex items-center justify-center gap-1.5 px-2.5 py-2 text-sm font-medium text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Resample Dataset
          </button>
        </div>
      </CollapsibleTile>

      {taskMode === 'regression' && (
        <CollapsibleTile title="Random Data Builder" icon={<FlaskConical className="w-4 h-4 text-accent" />}>
          <div className="grid grid-cols-2 gap-1.5 mb-2">
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.pattern === 'linear' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ pattern: 'linear' })}>Linear</button>
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.pattern === 'polynomial' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ pattern: 'polynomial' })}>Polynomial</button>
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.pattern === 'sinusoidal' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ pattern: 'sinusoidal' })}>Sinusoidal</button>
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.pattern === 'piecewise' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ pattern: 'piecewise' })}>Piecewise</button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 mb-2">
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.noiseType === 'gaussian' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ noiseType: 'gaussian' })}>Gaussian</button>
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.noiseType === 'heteroscedastic' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ noiseType: 'heteroscedastic' })}>Hetero</button>
            <button type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.noiseType === 'heavy_tail' ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ noiseType: 'heavy_tail' })}>Heavy Tail</button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 mb-3">
            {(['none', 'low', 'medium', 'high'] as const).map((level) => (
              <button key={level} type="button" className={`eval-chip min-h-9 whitespace-normal break-words leading-tight text-center text-[11px] sm:text-xs px-1.5 ${randomDataRecipe.outlierLevel === level ? 'eval-chip-active' : ''}`} onClick={() => setRandomDataRecipe({ outlierLevel: level })}>
                {level}
              </button>
            ))}
          </div>
          <Toggle label="Correlated Features" checked={randomDataRecipe.correlatedFeatures} onChange={(checked) => setRandomDataRecipe({ correlatedFeatures: checked })} />
          <button
            type="button"
            onClick={generateRandomDatasetFromRecipe}
            className="control-option interactive-lift w-full inline-flex items-center justify-center gap-1.5 px-2.5 py-2 mt-3 text-sm font-medium text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Generate and Switch
          </button>
        </CollapsibleTile>
      )}

      {taskMode === 'regression' && (dataset === 'outliers' || dataset === 'heteroscedastic') && (
        <div className="material-panel-soft p-2.5 flex items-start gap-2 border-amber-400/35 bg-amber-200/20 dark:bg-amber-400/10">
          <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-amber-700 dark:text-amber-200">
            This dataset stresses OLS assumptions. Compare regularized, polynomial, and stepwise variants for robustness.
          </p>
        </div>
      )}
    </div>
  );
}

export function ModelControlPanel() {
  const {
    taskMode,
    modelType,
    setModelType,
    params,
    setParam,
    showAssumptions,
    setShowAssumptions,
    showOlsSolution,
    setShowOlsSolution,
    compareWithOls,
    setCompareWithOls,
    showClassificationDiagnostics,
    setShowClassificationDiagnostics,
    resetParams,
    selectedMetrics,
    setSelectedMetrics,
    usesRegularization,
    usesL1Ratio,
    usesPolynomialDegree,
    usesStepwiseTerms,
  } = useControlStore();

  const regressionFamily: RegressionFamilyFilter = modelType === 'ols'
    ? 'linear'
    : modelType === 'ridge' || modelType === 'lasso' || modelType === 'elasticnet' || modelType === 'pcr_regressor' || modelType === 'pls_regressor' || modelType === 'svm_regressor'
      ? 'regularized'
      : 'nonlinear';
  const classificationFamily: ClassificationFamilyFilter = modelType === 'knn_classifier'
    ? 'local'
    : modelType === 'decision_tree_classifier' || modelType === 'random_forest_classifier'
      ? 'tree'
    : modelType === 'adaboost_classifier' || modelType === 'gradient_boosting_classifier'
      ? 'boosting'
    : modelType === 'svm_classifier'
      ? 'margin'
      : 'linear';
  const metricOptions = taskMode === 'classification' ? classificationMetricOptions : regressionMetricOptions;

  const setRegressionFamily = (family: RegressionFamilyFilter) => {
    const fallback = regressionModelOptions.find((option) => option.family === family);
    if (fallback) setModelType(fallback.value);
  };
  const setClassificationFamily = (family: ClassificationFamilyFilter) => {
    const fallback = classificationModelOptions.find((option) => option.family === family);
    if (fallback) setModelType(fallback.value);
  };
  const allowedModelOptions = taskMode === 'classification'
    ? classificationModelOptions
    : regressionModelOptions;
  const activeVisibleFamily = taskMode === 'classification'
    ? classificationFamily
    : regressionFamily;
  const toggleMetric = (metric: MetricKey) => {
    if (selectedMetrics.includes(metric)) {
      if (selectedMetrics.length === 1) return;
      setSelectedMetrics(selectedMetrics.filter((entry) => entry !== metric));
      return;
    }
    setSelectedMetrics([...selectedMetrics, metric]);
  };

  return (
    <div className="space-y-3">
      <CollapsibleTile title="Model Family" icon={<Filter className="w-4 h-4 text-accent" />} defaultOpen>
        <div className="flex flex-wrap gap-1.5 mb-3">
          {taskMode === 'classification' ? (
            <>
              <button type="button" onClick={() => setClassificationFamily('linear')} className={`model-family-chip ${activeVisibleFamily === 'linear' ? 'model-family-chip-active' : ''}`}>Linear</button>
              <button type="button" onClick={() => setClassificationFamily('local')} className={`model-family-chip ${activeVisibleFamily === 'local' ? 'model-family-chip-active' : ''}`}>Local</button>
              <button type="button" onClick={() => setClassificationFamily('margin')} className={`model-family-chip ${activeVisibleFamily === 'margin' ? 'model-family-chip-active' : ''}`}>Margin</button>
              <button type="button" onClick={() => setClassificationFamily('tree')} className={`model-family-chip ${activeVisibleFamily === 'tree' ? 'model-family-chip-active' : ''}`}>Tree/Ensemble</button>
              <button type="button" onClick={() => setClassificationFamily('boosting')} className={`model-family-chip ${activeVisibleFamily === 'boosting' ? 'model-family-chip-active' : ''}`}>Boosting</button>
            </>
          ) : (
            <>
              <button type="button" onClick={() => setRegressionFamily('linear')} className={`model-family-chip ${activeVisibleFamily === 'linear' ? 'model-family-chip-active' : ''}`}>Linear</button>
              <button type="button" onClick={() => setRegressionFamily('regularized')} className={`model-family-chip ${activeVisibleFamily === 'regularized' ? 'model-family-chip-active' : ''}`}>Regularized</button>
              <button type="button" onClick={() => setRegressionFamily('nonlinear')} className={`model-family-chip ${activeVisibleFamily === 'nonlinear' ? 'model-family-chip-active' : ''}`}>Nonlinear</button>
            </>
          )}
        </div>
        <div className="space-y-1.5">
          {allowedModelOptions
            .filter((option) => option.family === activeVisibleFamily)
            .map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setModelType(option.value)}
                className={`control-option model-filter-card interactive-lift w-full text-left px-3 py-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                  modelType === option.value ? 'control-option-active' : ''
                }`}
              >
                <div className="text-sm font-medium text-text-primary">{option.label}</div>
                <div className="text-xs leading-snug text-text-secondary mt-0.5">{option.description}</div>
              </button>
            ))}
        </div>
      </CollapsibleTile>

      <CollapsibleTile title="Scoreboard Metrics" icon={<BarChart3 className="w-4 h-4 text-accent" />}>
        <div className="flex items-center gap-2 mb-3">
          <h3 className="panel-title">Scoreboard Metrics</h3>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {metricOptions.map((metric) => (
            <button
              key={metric.key}
              type="button"
              onClick={() => toggleMetric(metric.key)}
              className={`eval-chip text-left ${selectedMetrics.includes(metric.key) ? 'eval-chip-active' : ''}`}
            >
              {metric.label}
            </button>
          ))}
        </div>
      </CollapsibleTile>

      <CollapsibleTile title="Hyperparameters" icon={<Sliders className="w-4 h-4 text-accent" />} defaultOpen>
        <div className="flex items-center gap-2 mb-3">
          <h3 className="panel-title">Hyperparameters</h3>
          {featureFlags.ff_info_microcards && (
            <InfoPopover
              label="Hyperparameters"
              what="These controls tune model complexity, regularization, neighborhood size, margins, and boosting behavior."
              why="Small parameter moves can strongly change generalization and confusion outcomes."
              tryNext="Change one slider at a time and watch validation metrics instead of training fit only."
            />
          )}
        </div>
        <div className="space-y-3.5">
          {taskMode === 'regression' && usesRegularization && (
            <Slider
              label="Alpha (Regularization)"
              value={params.alpha}
              min={0}
              max={3}
              step={0.01}
              onChange={(value) => setParam('alpha', value)}
            />
          )}
          {taskMode === 'regression' && usesL1Ratio && (
            <Slider
              label="L1 Ratio"
              value={params.l1Ratio}
              min={0}
              max={1}
              step={0.01}
              onChange={(value) => setParam('l1Ratio', value)}
            />
          )}
          {taskMode === 'regression' && usesPolynomialDegree && (
            <Slider
              label="Polynomial Degree"
              value={params.polynomialDegree}
              min={1}
              max={6}
              step={1}
              onChange={(value) => setParam('polynomialDegree', Math.round(value))}
            />
          )}
          {taskMode === 'regression' && usesStepwiseTerms && (
            <Slider
              label="Selected Terms"
              value={params.stepwiseTerms}
              min={1}
              max={Math.max(1, params.polynomialDegree)}
              step={1}
              onChange={(value) => setParam('stepwiseTerms', Math.round(value))}
            />
          )}
          {taskMode === 'classification' && modelType === 'knn_classifier' && (
            <Slider
              label="K (Neighbors)"
              value={params.knnK}
              min={1}
              max={25}
              step={1}
              onChange={(value) => setParam('knnK', Math.round(value))}
            />
          )}
          {taskMode === 'classification' && modelType === 'svm_classifier' && (
            <>
              <Slider
                label="SVM C"
                value={params.svmC}
                min={0.1}
                max={5}
                step={0.1}
                onChange={(value) => setParam('svmC', Number(value.toFixed(2)))}
              />
              <Slider
                label="SVM Gamma"
                value={params.svmGamma}
                min={0.1}
                max={5}
                step={0.1}
                onChange={(value) => setParam('svmGamma', Number(value.toFixed(2)))}
              />
            </>
          )}
          {taskMode === 'classification' && (modelType === 'decision_tree_classifier' || modelType === 'random_forest_classifier') && (
            <Slider
              label="Tree Depth"
              value={params.treeDepth}
              min={1}
              max={8}
              step={1}
              onChange={(value) => setParam('treeDepth', Math.round(value))}
            />
          )}
          {taskMode === 'classification' && modelType === 'random_forest_classifier' && (
            <Slider
              label="Forest Trees"
              value={params.forestTrees}
              min={5}
              max={100}
              step={1}
              onChange={(value) => setParam('forestTrees', Math.round(value))}
            />
          )}
          {taskMode === 'classification' && (modelType === 'adaboost_classifier' || modelType === 'gradient_boosting_classifier') && (
            <>
              <Slider
                label="Boosting Rounds"
                value={params.boostingRounds}
                min={5}
                max={180}
                step={1}
                onChange={(value) => setParam('boostingRounds', Math.round(value))}
              />
              <Slider
                label="Learning Rate"
                value={params.learningRate}
                min={0.02}
                max={1}
                step={0.01}
                onChange={(value) => setParam('learningRate', Number(value.toFixed(2)))}
              />
            </>
          )}
          {taskMode === 'regression' && modelType === 'svm_regressor' && (
            <>
              <Slider
                label="SVR C"
                value={params.svmC}
                min={0.1}
                max={6}
                step={0.1}
                onChange={(value) => setParam('svmC', Number(value.toFixed(2)))}
              />
              <Slider
                label="SVR Gamma"
                value={params.svmGamma}
                min={0.05}
                max={5}
                step={0.05}
                onChange={(value) => setParam('svmGamma', Number(value.toFixed(2)))}
              />
              <Slider
                label="SVR Epsilon"
                value={params.svmEpsilon ?? 0.1}
                min={0.01}
                max={0.8}
                step={0.01}
                onChange={(value) => setParam('svmEpsilon', Number(value.toFixed(2)))}
              />
            </>
          )}
          {taskMode === 'regression' && modelType === 'pcr_regressor' && (
            <Slider
              label="PCR Components"
              value={params.pcaComponents ?? 2}
              min={1}
              max={6}
              step={1}
              onChange={(value) => setParam('pcaComponents', Math.round(value))}
            />
          )}
          {taskMode === 'regression' && modelType === 'pls_regressor' && (
            <Slider
              label="PLS Components"
              value={params.plsComponents ?? 2}
              min={1}
              max={6}
              step={1}
              onChange={(value) => setParam('plsComponents', Math.round(value))}
            />
          )}
          {taskMode === 'classification' && (
            <Slider
              label="Decision Threshold"
              value={params.decisionThreshold}
              min={0.05}
              max={0.95}
              step={0.01}
              onChange={(value) => setParam('decisionThreshold', Number(value.toFixed(2)))}
            />
          )}
          <button
            type="button"
            onClick={resetParams}
            className="control-option interactive-lift w-full inline-flex items-center justify-center gap-1.5 px-2.5 py-2 text-sm font-medium text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Model Params
          </button>
        </div>
      </CollapsibleTile>

      <CollapsibleTile title="Diagnostics and Views" icon={<Settings2 className="w-4 h-4 text-accent" />}>
        <div className="flex items-center gap-2 mb-3">
          <h3 className="panel-title">Model Views</h3>
          {featureFlags.ff_info_microcards && (
            <InfoPopover
              label="Hyperparameter Views"
              what="These toggles control overlay and diagnostic lenses, not the underlying dataset."
              why="Keeping views mode-aware avoids dead controls and visual clutter."
              tryNext="Turn on diagnostics, then change one hyperparameter and observe metric + boundary shifts."
            />
          )}
        </div>
        <div className="space-y-2.5">
          {taskMode === 'regression' && <Toggle label="Overlay OLS Reference" checked={showOlsSolution} onChange={setShowOlsSolution} />}
          {taskMode === 'regression'
            ? <Toggle label="Assumption Diagnostics" checked={showAssumptions} onChange={setShowAssumptions} />
            : <Toggle label="Classification Diagnostics" checked={showClassificationDiagnostics} onChange={setShowClassificationDiagnostics} />}
          {taskMode === 'regression' && <Toggle label="Compare vs OLS Metrics" checked={compareWithOls} onChange={setCompareWithOls} />}
        </div>
      </CollapsibleTile>
    </div>
  );
}

export function ControlPanel() {
  return (
    <div className="space-y-3">
      <DataControlPanel />
      <ModelControlPanel />
    </div>
  );
}
