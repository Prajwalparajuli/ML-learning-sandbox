import { useEffect, useState } from 'react';
import { Info } from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';

interface InfoPopoverProps {
  label: string;
  what: string;
  why: string;
  tryNext: string;
  className?: string;
}

export function InfoPopover({ label, what, why, tryNext, className }: InfoPopoverProps) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);

  useEffect(() => {
    if (!open) {
      setPinned(false);
    }
  }, [open]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className={`info-dot ${className ?? ''}`}
          onMouseEnter={() => {
            if (!pinned) setOpen(true);
          }}
          onMouseLeave={() => {
            if (!pinned) setOpen(false);
          }}
          onClick={() => {
            setPinned((value) => !value);
            setOpen(true);
          }}
        >
          <Info className="w-3.5 h-3.5" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[280px] p-3">
        <p className="text-xs font-semibold text-text-primary mb-2">{label}</p>
        <div className="space-y-1.5 text-xs text-text-secondary">
          <p><span className="text-text-primary font-medium">What:</span> {what}</p>
          <p><span className="text-text-primary font-medium">Why:</span> {why}</p>
          <p><span className="text-text-primary font-medium">Try next:</span> {tryNext}</p>
        </div>
      </PopoverContent>
    </Popover>
  );
}

