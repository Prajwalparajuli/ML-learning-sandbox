import { useMemo, useState } from 'react';
import effectsData from '@/data/deep-learning/parameter_effects.json';
import type { ParameterEffectRecord } from '@/data/deep-learning/types';

type TabType = 'instant_visual' | 'precomputed_effect';

const records = effectsData as ParameterEffectRecord[];

export function DeepParameterLab() {
  const [tab, setTab] = useState<TabType>('instant_visual');
  const [selectedId, setSelectedId] = useState<string>(records[0]?.id ?? '');

  const filtered = useMemo(() => records.filter((record) => record.category === tab), [tab]);
  const selected = filtered.find((record) => record.id === selectedId) ?? filtered[0];

  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Parameter Lab</h3>
        <p className="deep-section-subtitle">Split parameters by interaction mode: instant visual vs precomputed training impact.</p>
      </div>

      <div className="deep-step-row">
        <button
          type="button"
          className={`deep-step-chip ${tab === 'instant_visual' ? 'deep-step-chip-active' : ''}`}
          onClick={() => {
            setTab('instant_visual');
            setSelectedId(records.find((record) => record.category === 'instant_visual')?.id ?? '');
          }}
        >
          Instant Visual
        </button>
        <button
          type="button"
          className={`deep-step-chip ${tab === 'precomputed_effect' ? 'deep-step-chip-active' : ''}`}
          onClick={() => {
            setTab('precomputed_effect');
            setSelectedId(records.find((record) => record.category === 'precomputed_effect')?.id ?? '');
          }}
        >
          Precomputed Impact
        </button>
      </div>

      <div className="deep-cnn-layout">
        <div className="deep-quiz-grid">
          {filtered.map((record) => (
            <button
              key={record.id}
              type="button"
              className={`deep-quiz-card ${selected?.id === record.id ? 'deep-quiz-card-active' : ''}`}
              onClick={() => setSelectedId(record.id)}
            >
              <p className="deep-quiz-question">{record.parameter}</p>
              <p className="deep-quiz-expl">{record.setting}</p>
            </button>
          ))}
        </div>
        {selected && (
          <article className="deep-quiz-card">
            <p className="deep-quiz-question">{selected.parameter}</p>
            <p className="deep-quiz-answer">{selected.setting}</p>
            <p className="deep-quiz-expl"><strong>Expected effect:</strong> {selected.expected_effect}</p>
            <p className="deep-quiz-expl"><strong>Visible signal:</strong> {selected.visual_signal}</p>
          </article>
        )}
      </div>
    </section>
  );
}
