import type { CnnPipelineStage } from '@/data/deep-learning/types';

interface CnnPipelineDiagramProps {
  stages: CnnPipelineStage[];
  activeStageId: CnnPipelineStage['id'];
  onStageChange: (id: CnnPipelineStage['id']) => void;
}

export function CnnPipelineDiagram({ stages, activeStageId, onStageChange }: CnnPipelineDiagramProps) {
  return (
    <section className="deep-panel deep-panel-primary">
      <div className="deep-section-head">
        <h4 className="deep-section-title">CNN Architecture Lane</h4>
        <p className="deep-section-subtitle">One active stage at a time. Click a stage to focus the workspace.</p>
      </div>
      <div className="deep-pipeline-lane">
        {stages.map((stage, index) => (
          <div key={stage.id} className="deep-pipeline-segment">
            <button
              type="button"
              className={`deep-stage-card ${activeStageId === stage.id ? 'deep-stage-card-active' : ''}`}
              onClick={() => onStageChange(stage.id)}
            >
              <span className="deep-stage-story">{stage.story_label}</span>
              <span className="deep-stage-label">{stage.label}</span>
            </button>
            {index < stages.length - 1 && <span className="deep-stage-arrow" aria-hidden>{'->'}</span>}
          </div>
        ))}
      </div>
    </section>
  );
}
