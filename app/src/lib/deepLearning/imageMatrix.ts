export async function loadGrayscaleMatrix(src: string, size = 8): Promise<number[][]> {
  const image = await loadImage(src);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Unable to create 2D canvas context');

  ctx.drawImage(image, 0, 0, size, size);
  const imageData = ctx.getImageData(0, 0, size, size).data;
  const matrix: number[][] = [];

  for (let row = 0; row < size; row += 1) {
    const values: number[] = [];
    for (let col = 0; col < size; col += 1) {
      const index = (row * size + col) * 4;
      const r = imageData[index];
      const g = imageData[index + 1];
      const b = imageData[index + 2];
      const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      values.push(Number(gray.toFixed(3)));
    }
    matrix.push(values);
  }

  return matrix;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    image.src = src;
  });
}
