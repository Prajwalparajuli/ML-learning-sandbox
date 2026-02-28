import type { DeepRuntimeBackend } from '@/data/deep-learning/types';

export interface TfjsRuntimeStatus {
  backend: DeepRuntimeBackend;
  ready: boolean;
  error: string | null;
  usingTfjs: boolean;
}

let cachedStatus: TfjsRuntimeStatus | null = null;

export async function initTfjsRuntime(): Promise<TfjsRuntimeStatus> {
  if (cachedStatus) return cachedStatus;

  const prefersWebgl = typeof window !== 'undefined' && !!window.WebGLRenderingContext;
  const backend: DeepRuntimeBackend = prefersWebgl ? 'webgl' : 'cpu';
  cachedStatus = {
    backend,
    ready: true,
    error: null,
    usingTfjs: false,
  };
  return cachedStatus;
}
