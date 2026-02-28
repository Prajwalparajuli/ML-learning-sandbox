import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import replayData from '@/data/deep-learning/training_replay.json';
import type { TrainingReplayPoint } from '@/data/deep-learning/types';

const points = replayData as TrainingReplayPoint[];

export function DeepTrainingReplay() {
  const [epoch, setEpoch] = useState<number>(10);

  const selected = useMemo(() => {
    return points.reduce((closest, current) => (
      Math.abs(current.epoch - epoch) < Math.abs(closest.epoch - epoch) ? current : closest
    ), points[0]);
  }, [epoch]);

  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Training Dynamics Replay</h3>
        <p className="deep-section-subtitle">Precomputed learning curves to show how training progresses without running live optimization.</p>
      </div>

      <label className="deep-control">
        <span>Epoch scrubber: {selected.epoch}</span>
        <input
          type="range"
          min={1}
          max={50}
          step={1}
          value={epoch}
          onChange={(event) => setEpoch(Number(event.target.value))}
        />
      </label>

      <div className="deep-chart-shell">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={points} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="training_loss" name="Train Loss" stroke="#f97316" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="validation_loss" name="Val Loss" stroke="#ef4444" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="training_accuracy" name="Train Acc" stroke="#22c55e" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="validation_accuracy" name="Val Acc" stroke="#14b8a6" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="deep-output-bars">
        <div className="deep-prob-row">
          <span>Cat</span>
          <div className="deep-prob-track"><span style={{ width: `${(selected.cat_confidence * 100).toFixed(1)}%` }} /></div>
          <strong>{(selected.cat_confidence * 100).toFixed(1)}%</strong>
        </div>
        <div className="deep-prob-row">
          <span>Dog</span>
          <div className="deep-prob-track"><span style={{ width: `${(selected.dog_confidence * 100).toFixed(1)}%` }} /></div>
          <strong>{(selected.dog_confidence * 100).toFixed(1)}%</strong>
        </div>
      </div>
    </section>
  );
}
