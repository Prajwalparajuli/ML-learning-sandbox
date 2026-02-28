import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import {
  estimateStrokeThickness,
  thinStrokeMatrix,
} from '@/lib/deepLearning/mnistPreprocess';

interface MnistDrawPadProps {
  image: number[][];
  onImageChange: (image: number[][]) => void;
  onPredict: () => void;
  centerStatus?: {
    isCentered: boolean;
    offsetX: number;
    offsetY: number;
    distance: number;
  } | null;
  predictionSummary?: {
    predictedClass: number;
    confidence: number;
    latencyMs: number;
  } | null;
}

const SIZE = 28;
const CANVAS_PX = 280;

const cloneImage = (image: number[][]): number[][] => image.map((row) => row.slice());

const makeEmptyImage = (): number[][] =>
  Array.from({ length: SIZE }, () => Array.from({ length: SIZE }, () => 0));

const shiftToCenter = (image: number[][]): number[][] => {
  let minR = SIZE;
  let minC = SIZE;
  let maxR = -1;
  let maxC = -1;

  for (let r = 0; r < SIZE; r += 1) {
    for (let c = 0; c < SIZE; c += 1) {
      if (image[r][c] > 0.12) {
        minR = Math.min(minR, r);
        minC = Math.min(minC, c);
        maxR = Math.max(maxR, r);
        maxC = Math.max(maxC, c);
      }
    }
  }

  if (maxR < 0 || maxC < 0) return image;

  const midR = (minR + maxR) / 2;
  const midC = (minC + maxC) / 2;
  const targetR = (SIZE - 1) / 2;
  const targetC = (SIZE - 1) / 2;
  const shiftR = Math.round(targetR - midR);
  const shiftC = Math.round(targetC - midC);

  const out = makeEmptyImage();
  for (let r = 0; r < SIZE; r += 1) {
    for (let c = 0; c < SIZE; c += 1) {
      const rr = r + shiftR;
      const cc = c + shiftC;
      if (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE) {
        out[rr][cc] = image[r][c];
      }
    }
  }
  return out;
};

export function MnistDrawPad({
  image,
  onImageChange,
  onPredict,
  centerStatus = null,
  predictionSummary = null,
}: MnistDrawPadProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const lastCellRef = useRef<{ row: number; col: number } | null>(null);
  const rafCommitRef = useRef<number | null>(null);
  const workingImageRef = useRef<number[][]>(cloneImage(image));
  const [strokeCount, setStrokeCount] = useState(0);
  const clearPad = () => {
    onImageChange(makeEmptyImage());
    setStrokeCount(0);
  };

  const filledPixels = useMemo(() => image.flat().filter((value) => value > 0.08).length, [image]);
  const thicknessScore = useMemo(() => estimateStrokeThickness(image), [image]);
  const isThickStroke = filledPixels > 0 && thicknessScore > 0.56;

  useEffect(() => {
    workingImageRef.current = cloneImage(image);
  }, [image]);

  useEffect(() => {
    return () => {
      if (rafCommitRef.current !== null) {
        window.cancelAnimationFrame(rafCommitRef.current);
      }
    };
  }, []);

  const commitImage = () => {
    if (rafCommitRef.current !== null) return;
    rafCommitRef.current = window.requestAnimationFrame(() => {
      rafCommitRef.current = null;
      onImageChange(cloneImage(workingImageRef.current));
    });
  };

  const paintAt = (row: number, col: number, boost = 1): void => {
    const next = workingImageRef.current;
    const brush = [
      { dr: 0, dc: 0, weight: 0.84 },
      { dr: -1, dc: 0, weight: 0.22 },
      { dr: 1, dc: 0, weight: 0.22 },
      { dr: 0, dc: -1, weight: 0.22 },
      { dr: 0, dc: 1, weight: 0.22 },
    ];

    for (let i = 0; i < brush.length; i += 1) {
      const rr = row + brush[i].dr;
      const cc = col + brush[i].dc;
      if (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE) {
        next[rr][cc] = Math.min(1, next[rr][cc] + brush[i].weight * boost);
      }
    }
    commitImage();
  };

  const paintSegment = (from: { row: number; col: number }, to: { row: number; col: number }) => {
    const dr = to.row - from.row;
    const dc = to.col - from.col;
    const steps = Math.max(Math.abs(dr), Math.abs(dc));
    if (steps <= 0) {
      paintAt(to.row, to.col, 1);
      return;
    }
    for (let step = 0; step <= steps; step += 1) {
      const row = Math.round(from.row + (dr * step) / steps);
      const col = Math.round(from.col + (dc * step) / steps);
      paintAt(row, col, 1);
    }
  };

  const pointerToCell = (event: ReactPointerEvent<HTMLCanvasElement>): { row: number; col: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;
    const col = Math.min(SIZE - 1, Math.floor((x / rect.width) * SIZE));
    const row = Math.min(SIZE - 1, Math.floor((y / rect.height) * SIZE));
    return { row, col };
  };

  const drawFromEvent = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const cell = pointerToCell(event);
    if (!cell) return;
    const previous = lastCellRef.current;
    if (!previous) {
      paintAt(cell.row, cell.col, 1);
    } else {
      paintSegment(previous, cell);
    }
    lastCellRef.current = cell;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#05070b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const pixelW = canvas.width / SIZE;
    const pixelH = canvas.height / SIZE;
    for (let r = 0; r < SIZE; r += 1) {
      for (let c = 0; c < SIZE; c += 1) {
        const value = image[r][c];
        if (value <= 0.001) continue;
        const alpha = Math.min(1, Math.max(0, value));
        ctx.fillStyle = `rgba(236, 242, 255, ${alpha.toFixed(3)})`;
        ctx.fillRect(c * pixelW, r * pixelH, pixelW, pixelH);
      }
    }

    ctx.strokeStyle = 'rgba(109, 124, 160, 0.2)';
    ctx.lineWidth = 0.5;
    for (let i = 1; i < SIZE; i += 1) {
      const x = i * pixelW;
      const y = i * pixelH;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
  }, [image]);

  return (
    <section className="deep-panel deep-panel-primary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">Input Canvas (MNIST 28x28)</h3>
        <p className="deep-section-subtitle">Draw one digit.</p>
      </div>
      <div className="deep-mnist-canvas-wrap">
        <canvas
          ref={canvasRef}
          width={CANVAS_PX}
          height={CANVAS_PX}
          className="deep-mnist-canvas"
          role="img"
          aria-label="MNIST drawing canvas"
          onPointerDown={(event) => {
            event.preventDefault();
            drawingRef.current = true;
            lastCellRef.current = null;
            workingImageRef.current = cloneImage(image);
            event.currentTarget.setPointerCapture(event.pointerId);
            drawFromEvent(event);
            setStrokeCount((value) => value + 1);
          }}
          onPointerMove={(event) => {
            event.preventDefault();
            if (!drawingRef.current) return;
            drawFromEvent(event);
          }}
          onPointerUp={(event) => {
            event.preventDefault();
            drawingRef.current = false;
            lastCellRef.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerLeave={() => {
            drawingRef.current = false;
            lastCellRef.current = null;
          }}
        />
      </div>
      <div className="deep-step-row">
        <button type="button" className="deep-step-chip" onClick={clearPad}>Clear</button>
        <button type="button" className="deep-step-chip" onClick={() => onImageChange(shiftToCenter(image))}>Center</button>
        <button type="button" className="deep-step-chip" onClick={() => onImageChange(thinStrokeMatrix(image))}>
          Thin Strokes
        </button>
        <button
          type="button"
          className="deep-step-chip deep-step-chip-primary"
          onClick={onPredict}
          title={isThickStroke ? 'Stroke is thick. You can still predict, or click Thin Strokes first.' : 'Predict digit'}
        >
          Predict
        </button>
      </div>
      {isThickStroke && (
        <p className="deep-feature-copy">
          Stroke is too thick for robust prediction. Click <strong>Thin Strokes</strong> before predict.
        </p>
      )}
      {centerStatus && !centerStatus.isCentered && filledPixels > 0 && (
        <p className="deep-feature-copy">
          Centering needed before predict. Offset: <strong>{centerStatus.offsetX.toFixed(1)}</strong>px x,{' '}
          <strong>{centerStatus.offsetY.toFixed(1)}</strong>px y.
        </p>
      )}
      {predictionSummary && (
        <p className="deep-feature-copy">
          Prediction: <strong>{predictionSummary.predictedClass}</strong> | Confidence:{' '}
          <strong>{(predictionSummary.confidence * 100).toFixed(2)}%</strong> | Latency:{' '}
          <strong>{predictionSummary.latencyMs.toFixed(2)} ms</strong>
        </p>
      )}
      <p className="deep-feature-copy">
        Active pixels: <strong>{filledPixels}</strong> | Strokes: <strong>{strokeCount}</strong> | Thickness:{' '}
        <strong>{(thicknessScore * 100).toFixed(0)}%</strong>
      </p>
    </section>
  );
}
