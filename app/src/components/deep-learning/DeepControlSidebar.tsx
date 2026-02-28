import type { DeepExperienceMode, DeepVizDensity } from '@/data/deep-learning/types';
import type { TfjsRuntimeStatus } from '@/lib/deepLearning/tfjsRuntime';

interface DeepControlSidebarProps {
  mode: DeepExperienceMode;
  setMode: (mode: DeepExperienceMode) => void;
  showLegacyMode: boolean;
  inferenceModel: 'mnist_mlp' | 'mnist_cnn';
  setInferenceModel: (model: 'mnist_mlp' | 'mnist_cnn') => void;
  density: DeepVizDensity;
  setDensity: (density: DeepVizDensity) => void;
  runtime: TfjsRuntimeStatus | null;
}

export function DeepControlSidebar({
  mode,
  setMode,
  showLegacyMode,
  inferenceModel,
  setInferenceModel,
  density,
  setDensity,
  runtime,
}: DeepControlSidebarProps) {
  return (
    <aside className="deep-sidebar">
      <section className="deep-panel deep-panel-sidebar">
        <h4 className="deep-section-title">Mode Switch</h4>
        <p className="deep-section-subtitle">Hard-separate model-derived inference from educational kernel simulation.</p>
        <div className="deep-sidebar-group">
          <div className="deep-toggle-grid">
            <button
              type="button"
              className={`deep-step-chip deep-step-chip-wide ${mode === 'real_inference' ? 'deep-step-chip-active' : ''}`}
              onClick={() => setMode('real_inference')}
            >
              Real Inference
            </button>
            <button
              type="button"
              className={`deep-step-chip deep-step-chip-wide ${mode === 'kernel_lab' ? 'deep-step-chip-active' : ''}`}
              onClick={() => setMode('kernel_lab')}
            >
              Kernel Lab
            </button>
            {showLegacyMode && (
              <button
                type="button"
                className={`deep-step-chip deep-step-chip-wide ${mode === 'legacy' ? 'deep-step-chip-active' : ''}`}
                onClick={() => setMode('legacy')}
              >
                Legacy
              </button>
            )}
          </div>
        </div>
      </section>

      {mode === 'real_inference' && (
        <section className="deep-panel deep-panel-sidebar">
          <h4 className="deep-section-title">Inference Controls</h4>
          <div className="deep-sidebar-group">
            <label className="deep-control">
              <span>Active model</span>
              <select value={inferenceModel} onChange={(event) => setInferenceModel(event.target.value as 'mnist_mlp' | 'mnist_cnn')}>
                <option value="mnist_mlp">MNIST MLP</option>
                <option value="mnist_cnn">MNIST CNN</option>
              </select>
            </label>
            <label className="deep-control">
              <span>Visualization density</span>
              <select value={density} onChange={(event) => setDensity(event.target.value as DeepVizDensity)}>
                <option value="balanced">Balanced (top-k=12)</option>
                <option value="high">High fidelity</option>
                <option value="minimal">Minimal/mobile-safe</option>
              </select>
            </label>
          </div>
          <div className="deep-sidebar-group">
            <p className="deep-feature-copy">
              <strong>Badge:</strong> Model-derived path
            </p>
            <p className="deep-feature-copy">
              Runtime backend: <strong>{runtime?.backend ?? 'initializing...'}</strong>
            </p>
            {!runtime?.usingTfjs && (
              <p className="deep-feature-copy">Running deterministic local fallback engine (offline-safe).</p>
            )}
          </div>
        </section>
      )}

      {mode === 'kernel_lab' && (
        <section className="deep-panel deep-panel-sidebar">
          <h4 className="deep-section-title">Kernel Lab Guardrail</h4>
          <p className="deep-feature-copy"><strong>Badge:</strong> Educational simulation</p>
          <p className="deep-feature-copy">Kernel edits do not modify real model weights or evaluation metrics.</p>
        </section>
      )}
    </aside>
  );
}
