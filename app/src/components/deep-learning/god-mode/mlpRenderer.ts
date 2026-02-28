import * as d3 from 'd3';
import { computeMlpLossAtWeights, lerp, weightColor } from './dummyData';
import type { GodModeAction, GodModeState } from './godModeTypes';

interface MlpRendererRefs {
  svg: SVGSVGElement;
  foldCanvas: HTMLCanvasElement;
  landscapeCanvas: HTMLCanvasElement;
}

interface EdgeDatum {
  id: 0 | 1;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface RendererActions {
  dispatch: (action: GodModeAction) => void;
}

const VIEW_BOX = { width: 860, height: 220 };

const POSITION = {
  input1: { x: 90, y: 70 },
  input2: { x: 90, y: 150 },
  sum: { x: 340, y: 110 },
  activation: { x: 520, y: 110 },
  output: { x: 730, y: 110 },
};

export class MlpRenderer {
  private readonly svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;

  private readonly edgeSelection: d3.Selection<SVGLineElement, EdgeDatum, SVGGElement, unknown>;

  private readonly weightLabelSelection: d3.Selection<SVGTextElement, EdgeDatum, SVGGElement, unknown>;

  private readonly sumValueText: d3.Selection<SVGTextElement, unknown, null, undefined>;

  private readonly outputValueText: d3.Selection<SVGTextElement, unknown, null, undefined>;

  private readonly foldCanvas: HTMLCanvasElement;

  private readonly landscapeCanvas: HTMLCanvasElement;

  private readonly actions: RendererActions;

  constructor(refs: MlpRendererRefs, actions: RendererActions) {
    this.actions = actions;
    this.foldCanvas = refs.foldCanvas;
    this.landscapeCanvas = refs.landscapeCanvas;

    this.svg = d3.select(refs.svg);
    this.svg.selectAll('*').remove();
    this.svg.attr('viewBox', `0 0 ${VIEW_BOX.width} ${VIEW_BOX.height}`);

    const edges: EdgeDatum[] = [
      { id: 0, x1: POSITION.input1.x, y1: POSITION.input1.y, x2: POSITION.sum.x, y2: POSITION.sum.y },
      { id: 1, x1: POSITION.input2.x, y1: POSITION.input2.y, x2: POSITION.sum.x, y2: POSITION.sum.y },
    ];

    const edgeGroup = this.svg.append('g').attr('class', 'studio-god-mlp-edges');

    const dragBehavior = d3
      .drag<SVGLineElement, EdgeDatum>()
      .on('start', (_, datum) => {
        this.actions.dispatch({ type: 'mlp/setDragEdge', edge: datum.id });
      })
      .on('drag', (event, datum) => {
        this.actions.dispatch({ type: 'mlp/dragWeight', edge: datum.id, deltaY: event.dy });
      })
      .on('end', () => {
        this.actions.dispatch({ type: 'mlp/setDragEdge', edge: null });
      });

    this.edgeSelection = edgeGroup
      .selectAll<SVGLineElement, EdgeDatum>('line')
      .data(edges)
      .join('line')
      .attr('x1', (datum) => datum.x1)
      .attr('y1', (datum) => datum.y1)
      .attr('x2', (datum) => datum.x2)
      .attr('y2', (datum) => datum.y2)
      .attr('class', 'studio-god-mlp-edge')
      .on('mouseenter', (_, datum) => {
        this.actions.dispatch({ type: 'mlp/setDragEdge', edge: datum.id });
      })
      .on('mouseleave', () => {
        this.actions.dispatch({ type: 'mlp/setDragEdge', edge: null });
      })
      .call(dragBehavior);

    this.weightLabelSelection = edgeGroup
      .selectAll<SVGTextElement, EdgeDatum>('text')
      .data(edges)
      .join('text')
      .attr('x', (datum) => (datum.x1 + datum.x2) / 2 - 12)
      .attr('y', (datum) => (datum.y1 + datum.y2) / 2 - 8)
      .attr('class', 'studio-god-mlp-weight');

    this.svg
      .append('line')
      .attr('x1', POSITION.activation.x + 44)
      .attr('y1', POSITION.activation.y)
      .attr('x2', POSITION.output.x - 24)
      .attr('y2', POSITION.output.y)
      .attr('class', 'studio-god-mlp-edge studio-god-mlp-edge-out');

    this.drawNodeCircle(POSITION.input1.x, POSITION.input1.y, 22, 'x1');
    this.drawNodeCircle(POSITION.input2.x, POSITION.input2.y, 22, 'x2');
    this.drawNodeCircle(POSITION.sum.x, POSITION.sum.y, 30, 'Sigma');
    this.drawActivationSquare(POSITION.activation.x - 34, POSITION.activation.y - 34, 68, 'f');
    this.drawNodeCircle(POSITION.output.x, POSITION.output.y, 26, 'y');

    this.svg.append('text').attr('x', POSITION.input1.x - 34).attr('y', 24).attr('class', 'studio-god-mlp-label').text('Input');
    this.svg.append('text').attr('x', POSITION.sum.x - 22).attr('y', 24).attr('class', 'studio-god-mlp-label').text('Summation');
    this.svg.append('text').attr('x', POSITION.activation.x - 25).attr('y', 24).attr('class', 'studio-god-mlp-label').text('Activation');
    this.svg.append('text').attr('x', POSITION.output.x - 18).attr('y', 24).attr('class', 'studio-god-mlp-label').text('Output');

    this.sumValueText = this.svg
      .append('text')
      .attr('x', POSITION.sum.x - 45)
      .attr('y', POSITION.sum.y + 54)
      .attr('class', 'studio-god-mlp-value');

    this.outputValueText = this.svg
      .append('text')
      .attr('x', POSITION.output.x - 50)
      .attr('y', POSITION.output.y + 50)
      .attr('class', 'studio-god-mlp-value');
  }

  render(state: GodModeState): void {
    const weights = state.mlp.w;

    this.edgeSelection
      .attr('stroke', (datum) => weightColor(weights[datum.id]))
      .attr('stroke-width', (datum) => (state.mlp.dragEdge === datum.id ? 7 : 4));

    this.weightLabelSelection
      .attr('fill', (datum) => weightColor(weights[datum.id]))
      .text((datum) => `w${datum.id + 1}=${weights[datum.id].toFixed(3)}`);

    this.sumValueText.text(`z=${state.mlp.z.toFixed(3)}`);
    this.outputValueText.text(`y=${state.mlp.y.toFixed(3)}  loss=${state.mlp.loss.toFixed(3)}`);

    this.drawFoldMap(state.mlp.foldT);
    this.drawLossLandscape(state);
  }

  private drawNodeCircle(x: number, y: number, radius: number, label: string): void {
    this.svg.append('circle').attr('cx', x).attr('cy', y).attr('r', radius).attr('class', 'studio-god-mlp-node-circle');
    this.svg.append('text').attr('x', x).attr('y', y + 5).attr('class', 'studio-god-mlp-node-text').text(label);
  }

  private drawActivationSquare(x: number, y: number, size: number, label: string): void {
    this.svg
      .append('rect')
      .attr('x', x)
      .attr('y', y)
      .attr('width', size)
      .attr('height', size)
      .attr('rx', 8)
      .attr('class', 'studio-god-mlp-activation-square');
    this.svg
      .append('text')
      .attr('x', x + size / 2)
      .attr('y', y + size / 2 + 5)
      .attr('class', 'studio-god-mlp-node-text')
      .text(label);
  }

  private drawFoldMap(foldT: number): void {
    const canvas = this.foldCanvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 8; i += 1) {
      const x = (i / 8) * width;
      const y = (i / 8) * height;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const mapX = (x: number) => ((x + 1) / 2) * width;
    const mapY = (y: number) => height - ((y + 1) / 2) * height;

    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 120; i += 1) {
      const x = -1 + (2 * i) / 120;
      const yRaw = x;
      const yRelu = Math.max(0, x);
      const y = lerp(yRaw, yRelu, foldT);
      const px = mapX(x);
      const py = mapY(y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.font = '11px Inter, Roboto, system-ui, sans-serif';
    ctx.fillText('ReLU manifold fold', 8, 16);
  }

  private drawLossLandscape(state: GodModeState): void {
    const canvas = this.landscapeCanvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, width, height);

    const project = (w1: number, w2: number, loss: number) => {
      const x = width * 0.5 + (w1 - w2) * 55;
      const y = height * 0.78 + (w1 + w2) * 24 - loss * 230;
      return { x, y };
    };

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
    ctx.lineWidth = 1;

    for (let r = -6; r <= 6; r += 1) {
      ctx.beginPath();
      for (let c = -6; c <= 6; c += 1) {
        const w1 = r / 4;
        const w2 = c / 4;
        const loss = computeMlpLossAtWeights(w1, w2, state.mlp.b, state.mlp.x, state.mlp.target);
        const point = project(w1, w2, loss);
        if (c === -6) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }
      ctx.stroke();

      ctx.beginPath();
      for (let c = -6; c <= 6; c += 1) {
        const w1 = c / 4;
        const w2 = r / 4;
        const loss = computeMlpLossAtWeights(w1, w2, state.mlp.b, state.mlp.x, state.mlp.target);
        const point = project(w1, w2, loss);
        if (c === -6) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }
      ctx.stroke();
    }

    const sourceBall = state.mlp.landscape.ball;
    const targetBall = state.mlp.landscape.targetBall;
    const t = state.mlp.landscape.active ? state.mlp.landscape.animT : 1;
    const currentW1 = lerp(sourceBall.w1, targetBall.w1, t);
    const currentW2 = lerp(sourceBall.w2, targetBall.w2, t);
    const currentLoss = computeMlpLossAtWeights(currentW1, currentW2, state.mlp.b, state.mlp.x, state.mlp.target);
    const ball = project(currentW1, currentW2, currentLoss);

    const gradient = ctx.createRadialGradient(ball.x, ball.y, 1, ball.x, ball.y, 16);
    gradient.addColorStop(0, 'rgba(250, 204, 21, 1)');
    gradient.addColorStop(1, 'rgba(250, 204, 21, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, 16, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#fde047';
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.font = '11px Inter, Roboto, system-ui, sans-serif';
    ctx.fillText('Loss Landscape (wireframe bowl)', 10, 16);
  }
}
