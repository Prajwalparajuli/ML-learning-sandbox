import type { DeepCheckpoint } from '@/data/deep-learning/types';

interface DeepCheckpointPanelProps {
  title: string;
  checkpoints: DeepCheckpoint[];
}

export function DeepCheckpointPanel({ title, checkpoints }: DeepCheckpointPanelProps) {
  const completed = checkpoints.filter((checkpoint) => checkpoint.completed).length;

  return (
    <section className="deep-panel deep-panel-sidebar">
      <div className="deep-section-head">
        <h4 className="deep-section-title">{title}</h4>
        <p className="deep-section-subtitle">{completed}/{checkpoints.length} completed</p>
      </div>
      <ul className="deep-checkpoint-list">
        {checkpoints.map((checkpoint) => (
          <li key={checkpoint.id} className={`deep-checkpoint-item ${checkpoint.completed ? 'deep-checkpoint-done' : ''}`}>
            <span aria-hidden>{checkpoint.completed ? '✓' : '○'}</span>
            <span>{checkpoint.label}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
