import { BlockMath } from 'react-katex';

interface FormulaDisplayProps {
  latex: string;
}

export function FormulaDisplay({ latex }: FormulaDisplayProps) {
  return (
    <div className="formula-paper flex items-center justify-center py-3 px-2.5">
      <BlockMath>{latex}</BlockMath>
    </div>
  );
}
