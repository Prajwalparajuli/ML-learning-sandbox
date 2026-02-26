export const Dropdown = ({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) => (
  <div className="flex flex-col mb-5">
    <label className="text-sm text-slate-300 mb-2 font-medium">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="p-3 border border-slate-600 rounded-lg bg-slate-800/80 text-slate-100 focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent transition-all"
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt.charAt(0).toUpperCase() + opt.slice(1)}
        </option>
      ))}
    </select>
  </div>
);
