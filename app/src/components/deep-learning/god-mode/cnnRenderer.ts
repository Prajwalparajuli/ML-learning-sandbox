import { clamp, lerp } from './dummyData';
import type { CnnStage, GodModeState } from './godModeTypes';

interface CnnRendererRefs {
  root: HTMLDivElement;
  inputGrid: HTMLDivElement;
  convGrid: HTMLDivElement;
  poolGrid: HTMLDivElement;
  flattenCol: HTMLDivElement;
  denseReadout: HTMLDivElement;
  kernelWindow: HTMLDivElement;
  frustumCanvas: HTMLCanvasElement;
  inputStage: HTMLDivElement;
  convStage: HTMLDivElement;
  poolStage: HTMLDivElement;
  flattenStage: HTMLDivElement;
  denseStage: HTMLDivElement;
}

const STAGE_RANK: Record<CnnStage, number> = {
  input: 0,
  conv: 1,
  pool: 2,
  flatten: 3,
  dense: 4,
};

export class CnnRenderer {
  private readonly refs: CnnRendererRefs;

  private readonly inputCells: HTMLSpanElement[];

  private readonly convCells: HTMLSpanElement[];

  private readonly poolCells: HTMLSpanElement[];

  private readonly flatCells: HTMLSpanElement[];

  constructor(refs: CnnRendererRefs) {
    this.refs = refs;
    this.inputCells = this.initCells(refs.inputGrid, 36);
    this.convCells = this.initCells(refs.convGrid, 16);
    this.poolCells = this.initCells(refs.poolGrid, 4);
    this.flatCells = this.initCells(refs.flattenCol, 4, true);
  }

  render(state: GodModeState): void {
    this.updateStages(state.cnn.stage);
    this.updateInput(state);
    this.updateConv(state);
    this.updatePool(state);
    this.updateFlatten(state);
    this.updateDense(state);
    this.drawFrustum(state);
  }

  private initCells(root: HTMLDivElement, count: number, text = false): HTMLSpanElement[] {
    root.replaceChildren();
    const cells: HTMLSpanElement[] = [];
    for (let i = 0; i < count; i += 1) {
      const span = document.createElement('span');
      if (text) {
        span.textContent = '0.00';
      }
      root.appendChild(span);
      cells.push(span);
    }
    return cells;
  }

  private updateStages(stage: CnnStage): void {
    this.refs.inputStage.classList.toggle('is-active', stage === 'input');
    this.refs.convStage.classList.toggle('is-active', stage === 'conv');
    this.refs.poolStage.classList.toggle('is-active', stage === 'pool');
    this.refs.flattenStage.classList.toggle('is-active', stage === 'flatten');
    this.refs.denseStage.classList.toggle('is-active', stage === 'dense');
  }

  private updateInput(state: GodModeState): void {
    const flat = state.cnn.input.flat();
    for (let i = 0; i < this.inputCells.length; i += 1) {
      const value = flat[i] ?? 0;
      const cell = this.inputCells[i];
      cell.style.opacity = `${0.25 + value * 0.75}`;
      cell.style.background = `rgba(37, 99, 235, ${0.2 + value * 0.8})`;
    }

    if (state.cnn.stage === 'conv') {
      this.refs.kernelWindow.style.opacity = '1';
      this.refs.kernelWindow.style.left = `${(state.cnn.scan.col / 6) * 100}%`;
      this.refs.kernelWindow.style.top = `${(state.cnn.scan.row / 6) * 100}%`;
    } else {
      this.refs.kernelWindow.style.opacity = '0';
    }
  }

  private updateConv(state: GodModeState): void {
    const flatConv = state.cnn.conv.flat();
    const visible = state.cnn.stage === 'input' ? 0 : state.cnn.stage === 'conv' ? state.cnn.step : 16;

    for (let i = 0; i < this.convCells.length; i += 1) {
      const cell = this.convCells[i];
      const isVisible = i < visible;
      const value = isVisible ? (flatConv[i] ?? 0) : 0;
      cell.style.opacity = isVisible ? '1' : '0.18';
      cell.style.background = `rgba(14, 165, 233, ${0.2 + clamp(value / 3, 0, 1) * 0.8})`;
      cell.textContent = value.toFixed(2);
    }
  }

  private updatePool(state: GodModeState): void {
    const flatPool = state.cnn.pool.flat();
    const visible = STAGE_RANK[state.cnn.stage] >= STAGE_RANK.pool;

    for (let i = 0; i < this.poolCells.length; i += 1) {
      const cell = this.poolCells[i];
      const value = visible ? (flatPool[i] ?? 0) : 0;
      cell.style.opacity = visible ? '1' : '0.22';
      cell.style.background = `rgba(16, 185, 129, ${0.2 + clamp(value / 3, 0, 1) * 0.8})`;
      cell.textContent = value.toFixed(2);
    }
  }

  private updateFlatten(state: GodModeState): void {
    const flat = state.cnn.flat;
    const visibleRank = STAGE_RANK[state.cnn.stage] >= STAGE_RANK.flatten;

    for (let i = 0; i < this.flatCells.length; i += 1) {
      const cell = this.flatCells[i];
      const value = flat[i] ?? 0;
      const baseT = visibleRank ? state.cnn.unzipT : 0;
      const t = clamp(baseT * 1.6 - i * 0.2, 0, 1);
      cell.style.opacity = `${0.12 + t * 0.88}`;
      cell.style.transform = `translateX(${lerp(-42, 0, t)}px)`;
      cell.style.color = t > 0.8 ? '#e2e8f0' : '#94a3b8';
      cell.style.borderColor = t > 0.8 ? 'rgba(148,163,184,0.7)' : 'rgba(148,163,184,0.3)';
      cell.textContent = value.toFixed(2);
    }
  }

  private updateDense(state: GodModeState): void {
    const visible = STAGE_RANK[state.cnn.stage] >= STAGE_RANK.dense;
    const edge = state.cnn.dense[0] ?? 0;
    const other = state.cnn.dense[1] ?? 0;

    this.refs.denseReadout.innerHTML = `
      <div class="studio-god-dense-pill ${visible && edge >= other ? 'is-win' : ''}">edge ${(edge * 100).toFixed(1)}%</div>
      <div class="studio-god-dense-pill ${visible && other > edge ? 'is-win' : ''}">other ${(other * 100).toFixed(1)}%</div>
    `;
    this.refs.denseReadout.style.opacity = visible ? '1' : '0.28';
  }

  private drawFrustum(state: GodModeState): void {
    const canvas = this.refs.frustumCanvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rootRect = this.refs.root.getBoundingClientRect();
    const inputRect = this.refs.inputGrid.getBoundingClientRect();
    const convRect = this.refs.convGrid.getBoundingClientRect();

    canvas.width = Math.max(1, Math.floor(rootRect.width));
    canvas.height = Math.max(1, Math.floor(rootRect.height));

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state.cnn.stage !== 'conv' || state.cnn.frustumT <= 0.02) return;

    const inCell = inputRect.width / 6;
    const outCell = convRect.width / 4;

    const left = inputRect.left - rootRect.left + state.cnn.scan.col * inCell;
    const top = inputRect.top - rootRect.top + state.cnn.scan.row * inCell;
    const right = left + inCell * 3;
    const bottom = top + inCell * 3;

    const targetX = convRect.left - rootRect.left + (state.cnn.scan.col + 0.5) * outCell;
    const targetY = convRect.top - rootRect.top + (state.cnn.scan.row + 0.5) * outCell;

    const corners = [
      { x: left, y: top },
      { x: right, y: top },
      { x: right, y: bottom },
      { x: left, y: bottom },
    ];

    const alpha = 0.08 + state.cnn.frustumT * 0.2;

    ctx.fillStyle = `rgba(56, 189, 248, ${alpha})`;
    for (let i = 0; i < corners.length; i += 1) {
      const next = corners[(i + 1) % corners.length];
      ctx.beginPath();
      ctx.moveTo(corners[i].x, corners[i].y);
      ctx.lineTo(next.x, next.y);
      ctx.lineTo(targetX, targetY);
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = `rgba(125, 211, 252, ${0.45 + state.cnn.frustumT * 0.4})`;
    ctx.lineWidth = 1.6;
    for (const corner of corners) {
      ctx.beginPath();
      ctx.moveTo(corner.x, corner.y);
      ctx.lineTo(targetX, targetY);
      ctx.stroke();
    }

    const glow = ctx.createRadialGradient(targetX, targetY, 2, targetX, targetY, 14);
    glow.addColorStop(0, 'rgba(125, 211, 252, 1)');
    glow.addColorStop(1, 'rgba(125, 211, 252, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(targetX, targetY, 14, 0, Math.PI * 2);
    ctx.fill();
  }
}
