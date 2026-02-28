import rawManifest from './index.json';
import type {
  DeepDefaults,
  DeepImageRecord,
  DeepLearningManifest,
  DeepLearningResolvedManifest,
  DeepResolvedImageRecord,
} from './types';

const assetModules = import.meta.glob('./assets/*.{jpg,jpeg,png,webp}', { eager: true, import: 'default' }) as Record<string, string>;

const isValidFilter = (value: unknown): value is DeepDefaults['default_filter'] =>
  value === 'horizontal_edge' || value === 'vertical_edge';

const isValidImage = (image: unknown): image is DeepImageRecord => {
  if (!image || typeof image !== 'object') return false;
  const candidate = image as Record<string, unknown>;
  return typeof candidate.id === 'number'
    && typeof candidate.key === 'string'
    && typeof candidate.label === 'string'
    && (candidate.class === 'cat' || candidate.class === 'dog')
    && typeof candidate.file === 'string';
};

function toManifest(value: unknown): DeepLearningManifest {
  if (!value || typeof value !== 'object') {
    throw new Error('Deep learning manifest must be an object.');
  }

  const candidate = value as Record<string, unknown>;
  const defaults = candidate.defaults as Record<string, unknown> | undefined;
  const images = candidate.images;

  if (!defaults || typeof defaults !== 'object') {
    throw new Error('Deep learning manifest is missing `defaults`.');
  }
  if (typeof defaults.sample_image_id !== 'number' || !isValidFilter(defaults.default_filter)) {
    throw new Error('Deep learning defaults are invalid.');
  }
  if (!Array.isArray(images) || images.length === 0 || !images.every(isValidImage)) {
    throw new Error('Deep learning manifest requires a valid non-empty `images` list.');
  }

  return {
    defaults: {
      sample_image_id: defaults.sample_image_id,
      default_filter: defaults.default_filter,
    },
    images: images.map((image) => ({ ...(image as DeepImageRecord) })),
  };
}

function resolveImageSrc(file: string): string {
  const key = `./${file.replace(/\\/g, '/')}`;
  const src = assetModules[key];
  if (!src) {
    throw new Error(`Deep learning asset not found for manifest file: ${file}`);
  }
  return src;
}

export function getDeepLearningManifest(): DeepLearningResolvedManifest {
  const manifest = toManifest(rawManifest);
  const resolvedImages: DeepResolvedImageRecord[] = manifest.images.map((image) => ({
    ...image,
    src: resolveImageSrc(image.file),
  }));

  const hasDefault = resolvedImages.some((image) => image.id === manifest.defaults.sample_image_id);
  if (!hasDefault) {
    throw new Error(`Deep learning default image id ${manifest.defaults.sample_image_id} was not found.`);
  }

  return {
    defaults: manifest.defaults,
    images: resolvedImages,
  };
}
