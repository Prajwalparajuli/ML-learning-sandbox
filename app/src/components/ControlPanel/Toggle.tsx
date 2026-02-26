import { useState } from 'react';

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export function Toggle({ label, checked, onChange }: ToggleProps) {
  const controlId = label.toLowerCase().replace(/\s+/g, '-');
  const [pressed, setPressed] = useState(false);

  return (
    <div className="flex items-center justify-between">
      <label htmlFor={controlId} className="text-sm text-text-secondary">{label}</label>
      <button
        id={controlId}
        type="button"
        role="switch"
        aria-checked={checked}
        onPointerDown={() => setPressed(true)}
        onPointerUp={() => setPressed(false)}
        onPointerLeave={() => setPressed(false)}
        onPointerCancel={() => setPressed(false)}
        onClick={() => onChange(!checked)}
        className={`premium-toggle relative w-11 h-6 rounded-full border cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 ${
          checked ? 'premium-toggle-on bg-accent border-accent/40' : 'premium-toggle-off bg-surface-muted border-border-subtle'
        } ${pressed ? 'premium-toggle-pressed' : ''}`}
      >
        <span
          className={`premium-toggle-knob absolute top-1 left-1 w-4 h-4 bg-white rounded-full ${
            checked ? 'translate-x-5' : 'translate-x-0'
          } ${pressed ? 'premium-toggle-knob-pressed' : ''}`}
        />
      </button>
    </div>
  );
}
