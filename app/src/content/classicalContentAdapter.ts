import type { DatasetType, EvaluationMode, MetricKey, ModelType } from '../store/modelStore';

type SourceStatus = 'verified' | 'needs-source-check';

export interface FamilyRecord {
  familyId: string;
  title: string;
  methods: string[];
}

export interface CitationRecord {
  topic: string;
  sourceTitle: string;
  chapterOrSection: string;
  pageRange: string;
  sourceStatus: SourceStatus;
}

const toTitle = (id: string): string =>
  id
    .split('_')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ');

export const classicalFamilies: FamilyRecord[] = [
  {
    familyId: 'regression',
    title: 'Regression',
    methods: [
      'ols_regression',
      'ridge_regression',
      'lasso_regression',
      'elastic_net',
      'polynomial_regression',
      'forward_backward_stepwise',
      'best_subset',
      'pcr',
      'pls',
      'poisson_glm',
    ],
  },
  {
    familyId: 'classification',
    title: 'Classification',
    methods: ['logistic_regression', 'multinomial_logistic', 'lda', 'qda', 'naive_bayes', 'knn', 'svm_classifier'],
  },
  {
    familyId: 'trees_ensembles',
    title: 'Trees & Ensembles',
    methods: ['decision_tree', 'bagging', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost'],
  },
  {
    familyId: 'nonlinear_additive',
    title: 'Nonlinear & Additive',
    methods: ['step_functions', 'regression_splines', 'smoothing_splines', 'loess', 'gams'],
  },
  {
    familyId: 'unsupervised',
    title: 'Unsupervised',
    methods: ['pca', 'kmeans', 'hierarchical_clustering', 'gmm'],
  },
  {
    familyId: 'resampling_selection',
    title: 'Resampling & Selection',
    methods: ['validation_set_approach', 'loocv', 'kfold_cv', 'bootstrap', 'stratified_kfold', 'nested_cv'],
  },
  {
    familyId: 'diagnostics_pitfalls',
    title: 'Diagnostics & Pitfalls',
    methods: ['qq_plot', 'scale_location_plot', 'leverage_cooks_distance', 'precision_recall_curve', 'calibration_curve', 'confusion_matrix'],
  },
];

export const modelContentMap: Record<
  ModelType,
  { title: string; familyId: string; methodId: string; explanation: string }
> = {
  ols: {
    title: 'OLS',
    familyId: 'regression',
    methodId: 'ols_regression',
    explanation:
      'Ordinary Least Squares minimizes residual sum of squares and provides a linear baseline for bias-variance comparisons.',
  },
  ridge: {
    title: 'Ridge',
    familyId: 'regression',
    methodId: 'ridge_regression',
    explanation:
      'Ridge adds an L2 penalty to stabilize coefficients under collinearity and typically improves generalization in noisy settings.',
  },
  lasso: {
    title: 'Lasso',
    familyId: 'regression',
    methodId: 'lasso_regression',
    explanation:
      'Lasso adds an L1 penalty that can shrink weaker coefficients to zero, giving sparse and interpretable models.',
  },
  elasticnet: {
    title: 'ElasticNet',
    familyId: 'regression',
    methodId: 'elastic_net',
    explanation:
      'Elastic Net blends L1 and L2 penalties, balancing feature selection with coefficient shrinkage for correlated predictors.',
  },
  polynomial: {
    title: 'Polynomial',
    familyId: 'regression',
    methodId: 'polynomial_regression',
    explanation:
      'Polynomial regression expands the basis (x, x^2, x^3, ...) so a linear solver can capture curved response patterns.',
  },
  forward_stepwise: {
    title: 'Forward Stepwise',
    familyId: 'regression',
    methodId: 'forward_backward_stepwise',
    explanation:
      'Forward stepwise selection adds candidate terms incrementally, targeting compact models with strong validation behavior.',
  },
  backward_stepwise: {
    title: 'Backward Stepwise',
    familyId: 'regression',
    methodId: 'forward_backward_stepwise',
    explanation:
      'Backward stepwise starts richer and removes weaker terms, often reducing variance while preserving fit quality.',
  },
  svm_regressor: {
    title: 'SVM Regressor',
    familyId: 'nonlinear_additive',
    methodId: 'svm_regression',
    explanation:
      'SVR uses margin-based regression with kernels to fit nonlinear trends while controlling complexity.',
  },
  pcr_regressor: {
    title: 'PCR',
    familyId: 'regression',
    methodId: 'pcr',
    explanation:
      'Principal Component Regression projects correlated predictors into orthogonal components before fitting regression.',
  },
  pls_regressor: {
    title: 'PLS',
    familyId: 'regression',
    methodId: 'pls',
    explanation:
      'Partial Least Squares learns latent factors aligned with target variation, useful with correlated high-dimensional inputs.',
  },
  logistic_classifier: {
    title: 'Logistic Classifier',
    familyId: 'classification',
    methodId: 'logistic_regression',
    explanation:
      'Logistic regression models class probability and separates classes with a linear decision boundary in feature space.',
  },
  knn_classifier: {
    title: 'KNN Classifier',
    familyId: 'classification',
    methodId: 'knn',
    explanation:
      'KNN predicts class by local neighborhood voting, making boundary flexibility depend on nearby examples.',
  },
  svm_classifier: {
    title: 'SVM Classifier',
    familyId: 'classification',
    methodId: 'svm_classifier',
    explanation:
      'SVM maximizes margin between classes, often yielding robust boundaries in overlapping regions.',
  },
  decision_tree_classifier: {
    title: 'Decision Tree',
    familyId: 'trees_ensembles',
    methodId: 'decision_tree',
    explanation:
      'Decision trees split feature space into rule-based regions, making local decisions easy to inspect.',
  },
  random_forest_classifier: {
    title: 'Random Forest',
    familyId: 'trees_ensembles',
    methodId: 'random_forest',
    explanation:
      'Random forests average many decorrelated trees to reduce variance and stabilize boundary behavior.',
  },
  adaboost_classifier: {
    title: 'AdaBoost',
    familyId: 'trees_ensembles',
    methodId: 'adaboost',
    explanation:
      'AdaBoost iteratively emphasizes difficult samples so weak learners combine into a stronger boundary.',
  },
  gradient_boosting_classifier: {
    title: 'Gradient Boosting',
    familyId: 'trees_ensembles',
    methodId: 'gradient_boosting',
    explanation:
      'Gradient boosting adds learners stage-by-stage to reduce residual error with controlled learning rate.',
  },
};

export const datasetExplanations: Record<DatasetType, string> = {
  linear: 'Near-linear signal for baseline assumption checking and interpretability.',
  noisy: 'Higher irreducible noise for bias-variance stress testing.',
  outliers: 'Anomalous points that magnify squared-loss sensitivity and robustness tradeoffs.',
  heteroscedastic: 'Residual variance changes with x, stressing constant-variance assumptions.',
  quadratic: 'Curved pattern where basis expansion reduces underfitting.',
  sinusoidal: 'Periodic pattern useful for checking approximation limits.',
  piecewise: 'Segmented local regimes where global linear fits underperform.',
  random_recipe: 'User-generated synthetic data from recipe controls for stress-testing model behavior.',
  class_linear: 'Linearly separable classes for baseline classification behavior.',
  class_overlap: 'Overlapping classes that stress precision/recall tradeoffs.',
  class_moons: 'Nonlinear interleaving shapes for boundary flexibility learning.',
  class_imbalanced: 'Skewed class frequency to expose metric sensitivity beyond accuracy.',
};

export const metricMeta: Record<MetricKey, { label: string; description: string; sourceStatus: SourceStatus }> = {
  r2: { label: 'R²', description: 'Fraction of variance explained by the model.', sourceStatus: 'verified' },
  adjustedR2: { label: 'Adjusted R²', description: 'R² corrected for predictor count.', sourceStatus: 'verified' },
  rmse: { label: 'RMSE', description: 'Root mean squared prediction error.', sourceStatus: 'verified' },
  mae: { label: 'MAE', description: 'Mean absolute prediction error.', sourceStatus: 'verified' },
  mse: { label: 'MSE', description: 'Mean squared prediction error.', sourceStatus: 'verified' },
  mape: { label: 'MAPE', description: 'Mean absolute percentage error.', sourceStatus: 'verified' },
  medianAe: { label: 'Median AE', description: 'Median absolute error.', sourceStatus: 'verified' },
  explainedVariance: { label: 'Explained Variance', description: 'Share of target variance captured by predictions.', sourceStatus: 'verified' },
  accuracy: { label: 'Accuracy', description: 'Overall fraction of correct class predictions.', sourceStatus: 'verified' },
  precision: { label: 'Precision', description: 'Among predicted positives, how many are truly positive.', sourceStatus: 'verified' },
  recall: { label: 'Recall', description: 'Among true positives, how many are recovered by the model.', sourceStatus: 'verified' },
  specificity: { label: 'Specificity', description: 'Among true negatives, how many are correctly rejected.', sourceStatus: 'verified' },
  f1: { label: 'F1', description: 'Harmonic mean of precision and recall.', sourceStatus: 'verified' },
  rocAuc: { label: 'ROC-AUC', description: 'Threshold-agnostic ranking quality across TPR/FPR tradeoff.', sourceStatus: 'verified' },
  prAuc: { label: 'PR-AUC', description: 'Precision-recall summary, especially informative for class imbalance.', sourceStatus: 'verified' },
  logLoss: { label: 'Log Loss', description: 'Penalizes incorrect and overconfident probability estimates.', sourceStatus: 'verified' },
};

export const diagnosticsMeta = [
  {
    id: 'qq_plot',
    title: 'Q-Q Plot',
    detail: 'Compares residual quantiles against normal quantiles to inspect tail behavior.',
    sourceStatus: 'verified' as const,
  },
  {
    id: 'scale_location_plot',
    title: 'Scale-Location Plot',
    detail: 'Shows spread of residual magnitude versus fitted values for variance stability checks.',
    sourceStatus: 'verified' as const,
  },
  {
    id: 'leverage_cooks_distance',
    title: "Leverage / Cook's Distance",
    detail: 'Identifies influential observations that can disproportionately alter fit.',
    sourceStatus: 'verified' as const,
  },
  {
    id: 'precision_recall_curve',
    title: 'Precision-Recall Curve',
    detail: 'Plots precision versus recall across thresholds.',
    sourceStatus: 'verified' as const,
  },
  {
    id: 'calibration_curve',
    title: 'Calibration Curve',
    detail: 'Compares predicted probabilities with observed frequencies.',
    sourceStatus: 'verified' as const,
  },
  {
    id: 'confusion_matrix',
    title: 'Confusion Matrix',
    detail: 'Summarizes TP, TN, FP, and FN counts for classification outcomes.',
    sourceStatus: 'verified' as const,
  },
];

export const resamplingMeta: Array<{
  id: string;
  title: string;
  detail: string;
  implementedModes: EvaluationMode[];
  sourceStatus: SourceStatus;
}> = [
  {
    id: 'validation_set_approach',
    title: 'Validation Set',
    detail: 'Single train/validation split for quick model comparison.',
    implementedModes: ['train_test'],
    sourceStatus: 'verified',
  },
  {
    id: 'loocv',
    title: 'LOOCV',
    detail: 'Special case of CV using one holdout sample per fold.',
    implementedModes: [],
    sourceStatus: 'verified',
  },
  {
    id: 'kfold_cv',
    title: 'K-Fold CV',
    detail: 'Averages validation results across K train/validation partitions.',
    implementedModes: ['cross_validation'],
    sourceStatus: 'verified',
  },
  {
    id: 'bootstrap',
    title: 'Bootstrap',
    detail: 'Resamples with replacement to estimate variability.',
    implementedModes: [],
    sourceStatus: 'verified',
  },
  {
    id: 'stratified_kfold',
    title: 'Stratified K-Fold',
    detail: 'Maintains class proportions per fold in classification tasks.',
    implementedModes: [],
    sourceStatus: 'verified',
  },
  {
    id: 'nested_cv',
    title: 'Nested CV',
    detail: 'Outer loop estimates generalization while inner loop tunes hyperparameters.',
    implementedModes: [],
    sourceStatus: 'verified',
  },
];

export const biasVarianceNarrative = {
  underfitting: 'High bias: both train and validation errors remain elevated because capacity is too low.',
  overfitting:
    'High variance: train error keeps dropping while validation error climbs after complexity passes the sweet spot.',
  sweetSpot:
    'The preferred operating point is where validation error is minimized while train/validation gap remains controlled.',
};

export const citationRecords: CitationRecord[] = [
  {
    topic: 'Validation Set and LOOCV',
    sourceTitle: 'ISLP',
    chapterOrSection: 'Chapter 5',
    pageRange: '202-205',
    sourceStatus: 'verified',
  },
  {
    topic: 'K-fold CV',
    sourceTitle: 'ISLP',
    chapterOrSection: 'Chapter 5',
    pageRange: '206-209',
    sourceStatus: 'verified',
  },
  {
    topic: 'Bootstrap',
    sourceTitle: 'ISLP',
    chapterOrSection: 'Chapter 5',
    pageRange: '212-214',
    sourceStatus: 'verified',
  },
  {
    topic: 'Polynomial boundary behavior',
    sourceTitle: 'ISLP',
    chapterOrSection: 'Chapter 7',
    pageRange: '290',
    sourceStatus: 'verified',
  },
  {
    topic: 'XGBoost mention',
    sourceTitle: 'Hands_On_ML_Geron.pdf',
    chapterOrSection: 'Chapter 7: Ensemble Learning and Random Forests',
    pageRange: '209',
    sourceStatus: 'verified',
  },
  {
    topic: 'Nested CV pattern',
    sourceTitle: 'ISLP',
    chapterOrSection: 'Chapter 6',
    pageRange: '278-279',
    sourceStatus: 'verified',
  },
  {
    topic: 'Precision-Recall AUC',
    sourceTitle: 'understanding-machine-learning-theory-algorithms.pdf',
    chapterOrSection: 'Chapter 17: Multiclass, Ranking, and Complex Prediction Problems',
    pageRange: '239-240',
    sourceStatus: 'verified',
  },
  {
    topic: 'Confusion Matrix',
    sourceTitle: 'Hands_On_ML_Geron.pdf',
    chapterOrSection: 'Chapter 3: Classification',
    pageRange: '92',
    sourceStatus: 'verified',
  },
  {
    topic: 'Precision-Recall Curve',
    sourceTitle: 'Hands_On_ML_Geron.pdf',
    chapterOrSection: 'Chapter 3: Classification',
    pageRange: '95-99',
    sourceStatus: 'verified',
  },
];

export const unresolvedNeedsSourceCheck: string[] = [];

export const sourceCheckLog = [
  {
    id: 'mape',
    source: 'scikit-learn',
    url: 'https://scikit-learn.org/1.2/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html',
  },
  {
    id: 'median_ae',
    source: 'scikit-learn',
    url: 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html',
  },
  {
    id: 'balanced_accuracy',
    source: 'scikit-learn',
    url: 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html',
  },
  {
    id: 'brier_score',
    source: 'scikit-learn',
    url: 'https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.brier_score_loss.html',
  },
  {
    id: 'calibration_curve',
    source: 'scikit-learn',
    url: 'https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html',
  },
  {
    id: 'qq_plot_scale_location_leverage_cooks',
    source: 'R stats + statsmodels',
    url: 'https://stat.ethz.ch/R-manual/R-devel/RHOME/library/stats/html/plot.lm.html',
  },
  {
    id: 'lightgbm',
    source: 'LightGBM docs + NeurIPS 2017',
    url: 'https://lightgbm.readthedocs.io/en/stable/Features.html',
  },
  {
    id: 'catboost',
    source: 'CatBoost docs + NeurIPS 2018',
    url: 'https://catboost.ai/docs/en/',
  },
];

export const methodLabel = (id: string): string => toTitle(id);
