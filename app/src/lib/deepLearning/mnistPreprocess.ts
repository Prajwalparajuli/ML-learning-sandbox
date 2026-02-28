export const MNIST_SIZE = 28;

export function clamp01(value: number): number {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

export function normalizeImageMatrix(image: number[][]): number[][] {
  return image.map((row) => row.map((value) => clamp01(value)));
}

export function estimateStrokeThickness(image: number[][]): number {
  const normalized = normalizeImageMatrix(image);
  let active = 0;
  let neighborTotal = 0;
  for (let r = 0; r < MNIST_SIZE; r += 1) {
    for (let c = 0; c < MNIST_SIZE; c += 1) {
      if (normalized[r][c] <= 0.1) continue;
      active += 1;
      let neighbors = 0;
      for (let dr = -1; dr <= 1; dr += 1) {
        for (let dc = -1; dc <= 1; dc += 1) {
          if (dr === 0 && dc === 0) continue;
          const rr = r + dr;
          const cc = c + dc;
          if (rr < 0 || rr >= MNIST_SIZE || cc < 0 || cc >= MNIST_SIZE) continue;
          if (normalized[rr][cc] > 0.1) neighbors += 1;
        }
      }
      neighborTotal += neighbors;
    }
  }

  if (active <= 0) return 0;
  const avgNeighbors = neighborTotal / active;
  const coverage = active / (MNIST_SIZE * MNIST_SIZE);
  const neighborTerm = (avgNeighbors / 8) * 0.75;
  const coverageTerm = Math.min(1, coverage / 0.22) * 0.25;
  return clamp01(neighborTerm + coverageTerm);
}

export function thinStrokeMatrix(image: number[][]): number[][] {
  let out = normalizeImageMatrix(image);
  for (let iter = 0; iter < 2; iter += 1) {
    const next = out.map((row) => row.slice());
    for (let r = 0; r < MNIST_SIZE; r += 1) {
      for (let c = 0; c < MNIST_SIZE; c += 1) {
        const value = out[r][c];
        if (value <= 0.08) continue;
        let neighborSum = 0;
        let neighborCount = 0;
        for (let dr = -1; dr <= 1; dr += 1) {
          for (let dc = -1; dc <= 1; dc += 1) {
            if (dr === 0 && dc === 0) continue;
            const rr = r + dr;
            const cc = c + dc;
            if (rr < 0 || rr >= MNIST_SIZE || cc < 0 || cc >= MNIST_SIZE) continue;
            neighborSum += out[rr][cc];
            neighborCount += 1;
          }
        }
        const localMean = neighborCount > 0 ? neighborSum / neighborCount : 0;
        const suppression = Math.max(0, localMean - 0.18) * 0.55;
        const reduced = clamp01(value - suppression);
        next[r][c] = Math.max(value * 0.32, reduced);
      }
    }
    out = next;
  }
  return out;
}

function bilinearSample(image: number[][], row: number, col: number): number {
  const rows = image.length;
  const cols = image[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return 0;

  const r0 = Math.floor(row);
  const c0 = Math.floor(col);
  const r1 = Math.min(rows - 1, r0 + 1);
  const c1 = Math.min(cols - 1, c0 + 1);
  const dr = row - r0;
  const dc = col - c0;

  const v00 = image[Math.max(0, Math.min(rows - 1, r0))][Math.max(0, Math.min(cols - 1, c0))];
  const v01 = image[Math.max(0, Math.min(rows - 1, r0))][c1];
  const v10 = image[r1][Math.max(0, Math.min(cols - 1, c0))];
  const v11 = image[r1][c1];

  const top = v00 * (1 - dc) + v01 * dc;
  const bottom = v10 * (1 - dc) + v11 * dc;
  return clamp01(top * (1 - dr) + bottom * dr);
}

export function preprocessDrawnDigit(image: number[][]): number[][] {
  const normalized = normalizeImageMatrix(image);
  const thinned = thinStrokeMatrix(normalized);
  const maxValue = Math.max(1e-6, ...thinned.flat());
  const boosted = thinned.map((row) => row.map((value) => clamp01(value / maxValue)));

  let minR = MNIST_SIZE;
  let minC = MNIST_SIZE;
  let maxR = -1;
  let maxC = -1;
  for (let r = 0; r < MNIST_SIZE; r += 1) {
    for (let c = 0; c < MNIST_SIZE; c += 1) {
      if (boosted[r][c] > 0.08) {
        minR = Math.min(minR, r);
        minC = Math.min(minC, c);
        maxR = Math.max(maxR, r);
        maxC = Math.max(maxC, c);
      }
    }
  }

  if (maxR < 0 || maxC < 0) {
    return normalizeImageMatrix(image);
  }

  const cropH = maxR - minR + 1;
  const cropW = maxC - minC + 1;
  const targetInner = 20;
  const scale = targetInner / Math.max(cropH, cropW);
  const resizedH = Math.max(1, Math.round(cropH * scale));
  const resizedW = Math.max(1, Math.round(cropW * scale));

  const resized = Array.from({ length: resizedH }, () => Array.from({ length: resizedW }, () => 0));
  for (let r = 0; r < resizedH; r += 1) {
    for (let c = 0; c < resizedW; c += 1) {
      const srcR = minR + ((r + 0.5) / resizedH) * cropH - 0.5;
      const srcC = minC + ((c + 0.5) / resizedW) * cropW - 0.5;
      resized[r][c] = bilinearSample(boosted, srcR, srcC);
    }
  }

  const out = Array.from({ length: MNIST_SIZE }, () => Array.from({ length: MNIST_SIZE }, () => 0));
  const startR = Math.floor((MNIST_SIZE - resizedH) / 2);
  const startC = Math.floor((MNIST_SIZE - resizedW) / 2);
  for (let r = 0; r < resizedH; r += 1) {
    for (let c = 0; c < resizedW; c += 1) {
      const rr = startR + r;
      const cc = startC + c;
      if (rr >= 0 && rr < MNIST_SIZE && cc >= 0 && cc < MNIST_SIZE) {
        out[rr][cc] = resized[r][c];
      }
    }
  }

  return centerImageByMass(out);
}

export function centerImageByMass(image: number[][]): number[][] {
  let mass = 0;
  let rowSum = 0;
  let colSum = 0;

  for (let r = 0; r < image.length; r += 1) {
    for (let c = 0; c < image[r].length; c += 1) {
      const value = clamp01(image[r][c]);
      mass += value;
      rowSum += r * value;
      colSum += c * value;
    }
  }

  if (mass <= 1e-6) return normalizeImageMatrix(image);

  const centerR = rowSum / mass;
  const centerC = colSum / mass;
  const target = (MNIST_SIZE - 1) / 2;
  const shiftR = Math.round(target - centerR);
  const shiftC = Math.round(target - centerC);

  const out = Array.from({ length: MNIST_SIZE }, () => Array.from({ length: MNIST_SIZE }, () => 0));
  for (let r = 0; r < MNIST_SIZE; r += 1) {
    for (let c = 0; c < MNIST_SIZE; c += 1) {
      const rr = r - shiftR;
      const cc = c - shiftC;
      if (rr >= 0 && rr < MNIST_SIZE && cc >= 0 && cc < MNIST_SIZE) {
        out[r][c] = clamp01(image[rr][cc]);
      }
    }
  }
  return out;
}

export function flattenImageMatrix(image: number[][]): number[] {
  const values: number[] = [];
  for (let r = 0; r < image.length; r += 1) {
    for (let c = 0; c < image[r].length; c += 1) values.push(clamp01(image[r][c]));
  }
  return values;
}

export function reshapeFlatImage(values: ArrayLike<number>, rows = MNIST_SIZE, cols = MNIST_SIZE): number[][] {
  const image: number[][] = [];
  for (let r = 0; r < rows; r += 1) {
    const row: number[] = [];
    for (let c = 0; c < cols; c += 1) {
      row.push(clamp01(values[r * cols + c] ?? 0));
    }
    image.push(row);
  }
  return image;
}

export function decodeBase64ToU8(base64: string): Uint8Array {
  if (typeof atob === 'function') {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  const fromNode = (globalThis as typeof globalThis & { Buffer?: { from: (value: string, encoding: string) => Uint8Array } }).Buffer;
  if (fromNode) return fromNode.from(base64, 'base64');
  return new Uint8Array();
}

export function matrixFromPackedU8(
  packed: Uint8Array,
  sampleIndex: number,
  shape: [number, number, number]
): number[][] {
  const [rows, cols] = shape;
  const pixelsPerSample = rows * cols;
  const start = sampleIndex * pixelsPerSample;
  const out: number[][] = [];
  for (let r = 0; r < rows; r += 1) {
    const row: number[] = [];
    for (let c = 0; c < cols; c += 1) {
      const raw = packed[start + r * cols + c] ?? 0;
      row.push(raw / 255);
    }
    out.push(row);
  }
  return out;
}

export function groupInto16(image: number[][]): number[] {
  const groups = Array.from({ length: 16 }, () => 0);
  const counts = Array.from({ length: 16 }, () => 0);
  for (let r = 0; r < image.length; r += 1) {
    for (let c = 0; c < image[r].length; c += 1) {
      const gr = Math.min(3, Math.floor(r / 7));
      const gc = Math.min(3, Math.floor(c / 7));
      const idx = gr * 4 + gc;
      groups[idx] += image[r][c];
      counts[idx] += 1;
    }
  }
  return groups.map((value, index) => (counts[index] > 0 ? value / counts[index] : 0));
}

export function convValid(input: number[][], kernel: number[][]): number[][] {
  const outRows = input.length - kernel.length + 1;
  const outCols = input[0].length - kernel[0].length + 1;
  const out: number[][] = Array.from({ length: outRows }, () => Array.from({ length: outCols }, () => 0));
  for (let r = 0; r < outRows; r += 1) {
    for (let c = 0; c < outCols; c += 1) {
      let sum = 0;
      for (let kr = 0; kr < kernel.length; kr += 1) {
        for (let kc = 0; kc < kernel[kr].length; kc += 1) {
          sum += input[r + kr][c + kc] * kernel[kr][kc];
        }
      }
      out[r][c] = sum;
    }
  }
  return out;
}

export function reluMatrix(matrix: number[][]): number[][] {
  return matrix.map((row) => row.map((value) => (value > 0 ? value : 0)));
}

export function maxPool2x2(matrix: number[][]): number[][] {
  const rows = Math.floor(matrix.length / 2);
  const cols = Math.floor(matrix[0].length / 2);
  const out: number[][] = Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const values = [
        matrix[r * 2][c * 2],
        matrix[r * 2][c * 2 + 1],
        matrix[r * 2 + 1][c * 2],
        matrix[r * 2 + 1][c * 2 + 1],
      ];
      out[r][c] = Math.max(...values);
    }
  }
  return out;
}

export function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const sum = exp.reduce((acc, value) => acc + value, 0);
  return exp.map((value) => value / Math.max(sum, Number.EPSILON));
}

export function argMax(values: number[]): number {
  let bestIndex = 0;
  let bestValue = values[0] ?? Number.NEGATIVE_INFINITY;
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > bestValue) {
      bestIndex = i;
      bestValue = values[i];
    }
  }
  return bestIndex;
}

export function topKIndices(values: number[], k: number): number[] {
  return values
    .map((value, index) => ({ value: Math.abs(value), index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, Math.max(1, k))
    .map((item) => item.index);
}
