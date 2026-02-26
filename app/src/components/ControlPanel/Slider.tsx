import { useEffect, useRef, useState } from 'react';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

export function Slider({ label, value, min, max, step, onChange }: SliderProps) {
  const controlId = label.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const range = Math.max(max - min, 1e-6);
  const progress = Math.min(100, Math.max(0, ((value - min) / range) * 100));
  const [active, setActive] = useState(false);
  const [draft, setDraft] = useState(value);
  const debounceRef = useRef<number | null>(null);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  useEffect(() => () => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
  }, []);

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label htmlFor={controlId} className="text-sm text-text-secondary">{label}</label>
        <span className={`text-sm font-mono text-accent bg-accent/12 border border-accent/20 px-2 py-0.5 rounded-lg transition-transform duration-150 ${active ? 'scale-105' : ''}`}>
          {draft.toFixed(step < 0.1 ? 2 : 1)}
        </span>
      </div>
      <input
        id={controlId}
        type="range"
        min={min}
        max={max}
        step={step}
        value={draft}
        onPointerDown={() => setActive(true)}
        onPointerUp={() => {
          setActive(false);
          onChange(draft);
        }}
        onPointerCancel={() => setActive(false)}
        onPointerLeave={() => setActive(false)}
        onChange={(e) => {
          const next = parseFloat(e.target.value);
          setDraft(next);
          if (!active) {
            onChange(next);
            return;
          }
          if (debounceRef.current) window.clearTimeout(debounceRef.current);
          debounceRef.current = window.setTimeout(() => onChange(next), 100);
        }}
        className="premium-range w-full accent-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 rounded-md"
        style={{ ['--range-progress' as any]: `${progress}%` }}
      />
      <div className="flex justify-between text-xs text-text-tertiary">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
