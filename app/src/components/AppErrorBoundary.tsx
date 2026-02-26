import React from 'react';

interface AppErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class AppErrorBoundary extends React.Component<React.PropsWithChildren, AppErrorBoundaryState> {
  constructor(props: React.PropsWithChildren) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { hasError: true, message: error?.message ?? 'Unknown runtime error' };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('AppErrorBoundary caught:', error, info);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-background text-foreground">
        <div className="max-w-2xl w-full rounded-2xl border border-red-400/35 bg-red-500/10 p-4 space-y-2">
          <h1 className="text-lg font-semibold">Runtime error</h1>
          <p className="text-sm text-text-secondary">
            The app crashed during render. Open browser console for full stack trace.
          </p>
          <pre className="text-xs whitespace-pre-wrap break-words bg-black/30 rounded-lg p-3">
            {this.state.message}
          </pre>
          <p className="text-xs text-text-tertiary">
            Try hard refresh (Ctrl+Shift+R) after restarting the dev server.
          </p>
        </div>
      </div>
    );
  }
}
