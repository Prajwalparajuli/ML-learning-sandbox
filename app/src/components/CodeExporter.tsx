import { useState } from 'react';
import { useModelStore } from '../store/modelStore';
import { generatePythonCode } from '../lib/dataUtils';
import { Copy, Check, FileCode } from 'lucide-react';
import { toast } from 'sonner';

export function CodeExporter() {
  const { modelType, params, data, evaluationMode, testRatio, cvFolds } = useModelStore();
  const [copied, setCopied] = useState(false);

  const featureCount = data[0]?.features.length ?? 1;
  const modelCode = generatePythonCode(modelType, params, evaluationMode, testRatio, cvFolds, featureCount);
  const dataX = data.map((point) => point.features.map((value) => Number(value.toFixed(4))));
  const dataY = data.map((point) => Number(point.y.toFixed(4)));
  const code = `# Reproducible data snapshot exported from ML Learning Sandbox
data_x = ${JSON.stringify(dataX)}
data_y = ${JSON.stringify(dataY)}

${modelCode}`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error('Clipboard access failed', {
        description: 'Copy and paste from the code block manually.',
      });
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <FileCode className="w-4 h-4 text-text-tertiary" />
          <span className="text-xs text-text-secondary">
            scikit-learn / {modelType.toUpperCase()}
          </span>
        </div>
        <button
          onClick={handleCopy}
          className="control-option interactive-lift flex items-center gap-1.5 px-3 py-1.5 text-accent text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              Copy
            </>
          )}
        </button>
      </div>
      
      <div className="code-block overflow-hidden">
        <pre className="p-3 text-xs font-mono text-text-primary overflow-x-auto whitespace-pre leading-snug">
          {code}
        </pre>
      </div>
    </div>
  );
}
