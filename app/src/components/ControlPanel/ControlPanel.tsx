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

export function DataControlPanel() {
  const {
    taskMode,
    modelType,
    setModelType,
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
  } = useControlStore();
  const datasetOptions = taskMode === 'classification' ? classificationDatasetOptions : regressionDatasetOptions;

  return (
    <div className="space-y-3">
      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-accent" />
          <h3 className="panel-title">Data Mood</h3>
        </div>
        {taskMode === 'regression' ? (
          <div className="grid grid-cols-2 gap-1.5 mb-2.5">
            <button type="button" onClick={() => setFeatureMode('1d')} className={`eval-chip ${featureMode === '1d' ? 'eval-chip-active' : ''}`}>1D Features</button>
            <button
              type="button"
              onClick={() => {
                setFeatureMode('2d');
                if (modelType === 'polynomial' || modelType === 'forward_stepwise' || modelType === 'backward_stepwise') {
                  setModelType('ridge');
                }
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
      </div>

      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-accent" />
          <h3 className="panel-title">Train & Validate</h3>
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
      </div>

      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <FlaskConical className="w-4 h-4 text-accent" />
          <h3 className="panel-title">Data Generation</h3>
        </div>
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
      </div>

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
    featureMode,
    selectedMetrics,
    setSelectedMetrics,
    usesRegularization,
    usesL1Ratio,
    usesPolynomialDegree,
    usesStepwiseTerms,
  } = useControlStore();

  const regressionFamily: RegressionFamilyFilter = modelType === 'ols'
    ? 'linear'
    : modelType === 'ridge' || modelType === 'lasso' || modelType === 'elasticnet'
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
    const fallback = regressionModelOptions.find((option) => option.family === family && (featureMode === '1d' || family !== 'nonlinear'));
    if (fallback) setModelType(fallback.value);
  };
  const setClassificationFamily = (family: ClassificationFamilyFilter) => {
    const fallback = classificationModelOptions.find((option) => option.family === family);
    if (fallback) setModelType(fallback.value);
  };
  const allowedModelOptions = taskMode === 'classification'
    ? classificationModelOptions
    : regressionModelOptions.filter((option) => featureMode === '1d' || option.family !== 'nonlinear');
  const activeVisibleFamily = taskMode === 'classification'
    ? classificationFamily
    : (featureMode === '2d' && regressionFamily === 'nonlinear' ? 'regularized' : regressionFamily);
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
      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <Filter className="w-4 h-4 text-accent" />
          <h3 className="panel-title">Model Family</h3>
        </div>
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
              <button type="button" onClick={() => setRegressionFamily('nonlinear')} disabled={featureMode === '2d'} className={`model-family-chip ${activeVisibleFamily === 'nonlinear' ? 'model-family-chip-active' : ''} ${featureMode === '2d' ? 'opacity-50 cursor-not-allowed' : ''}`}>Nonlinear</button>
            </>
          )}
        </div>
        {taskMode === 'regression' && featureMode === '2d' && (
          <p className="text-[11px] text-text-tertiary mb-2">Polynomial and stepwise variants are 1D-only in this view.</p>
        )}
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
      </div>

      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-accent" />
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
      </div>

      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <Sliders className="w-4 h-4 text-accent" />
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
      </div>

      <div className="material-panel p-3.5">
        <div className="flex items-center gap-2 mb-3">
          <Settings2 className="w-4 h-4 text-accent" />
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
      </div>
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
