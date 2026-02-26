import type { Data, Layout, Config } from 'plotly.js';

declare global {
  namespace Plotly {
    type Data = Data;
    type Layout = Layout;
    type Config = Config;
  }
}

export {};
