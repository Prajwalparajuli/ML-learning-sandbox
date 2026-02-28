import type { DeepResolvedImageRecord } from '@/data/deep-learning/types';

interface DeepImageSelectorProps {
  images: DeepResolvedImageRecord[];
  selectedImageId: number;
  onSelect: (id: number) => void;
}

export function DeepImageSelector({ images, selectedImageId, onSelect }: DeepImageSelectorProps) {
  return (
    <div className="deep-image-strip" role="listbox" aria-label="Select a cat or dog example image">
      {images.map((image) => {
        const active = image.id === selectedImageId;
        return (
          <button
            key={image.id}
            type="button"
            role="option"
            aria-selected={active}
            className={`deep-image-chip ${active ? 'deep-image-chip-active' : ''}`}
            onClick={() => onSelect(image.id)}
          >
            <img src={image.src} alt={image.label} className="deep-image-chip-thumb" loading="lazy" />
            <span className="deep-image-chip-label">{image.label}</span>
          </button>
        );
      })}
    </div>
  );
}
