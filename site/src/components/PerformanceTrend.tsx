import React from 'react';
import { HistoricalData, HistoricalScoreEntry } from '../types';
import './PerformanceTrend.css';

interface PerformanceTrendProps {
  data: HistoricalData;
}

const BENCH_KEYS: (keyof HistoricalScoreEntry)[] = [
  'practical_knowledge',
  'ifeval',
  'gsm8k',
  'creative_technical',
];

const LINE_COLORS = ['#06b6d4', '#fbbf24', '#a78bfa', '#34d399', '#f87171', '#f472b6'];

const shortModel = (m: string) => m.split('/').pop() || m;

/** Overall (composite) score for one historical entry: mean of the benchmarks present. */
function overallOf(entry: HistoricalScoreEntry): number | null {
  const vals = BENCH_KEYS.map((k) => entry[k]).filter((v): v is number => typeof v === 'number');
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

/**
 * Multi-line chart of each model's overall score over time, using the smoothed `rolling`
 * series when available. Answers "which models are performing better over time" at a glance.
 */
const PerformanceTrend: React.FC<PerformanceTrendProps> = ({ data }) => {
  const models = Object.entries(data.models);
  if (!models.length) return null;

  // Build each model's overall-over-time series (smoothed series preferred).
  const series = models.map(([name, m], i) => {
    const entries = m.rolling ?? m.scores;
    const points = entries
      .map((e) => ({ date: e.date, value: overallOf(e) }))
      .filter((p): p is { date: string; value: number } => p.value !== null);
    return { name, color: LINE_COLORS[i % LINE_COLORS.length], points };
  }).filter((s) => s.points.length > 0);

  if (!series.length) return null;

  // Shared x-axis: sorted union of all dates across models.
  const dates = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.date)))).sort();
  if (dates.length < 2) return null;
  const xIndex = new Map(dates.map((d, i) => [d, i]));

  const W = 720;
  const H = 340;
  const pad = { top: 20, right: 150, bottom: 36, left: 38 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const x = (date: string) => pad.left + (xIndex.get(date)! / (dates.length - 1)) * plotW;
  const y = (v: number) => pad.top + (1 - v) * plotH;

  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  // Label a handful of x dates to avoid crowding.
  const xLabelIdx = [0, Math.floor((dates.length - 1) / 2), dates.length - 1];

  return (
    <section className="performance-trend">
      <h2>Performance Over Time</h2>
      <p className="pt-sub">
        Each model's overall score (composite of its benchmarks), smoothed across weekly
        runs. Lines crossing = the lead changed hands.
      </p>

      <div className="pt-chart-wrap">
        <svg className="pt-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Model performance over time">
          {/* gridlines + y labels */}
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={pad.left} y1={y(t)} x2={pad.left + plotW} y2={y(t)} className="pt-grid" />
              <text x={pad.left - 6} y={y(t) + 3} className="pt-axis" textAnchor="end">
                {t.toFixed(2)}
              </text>
            </g>
          ))}

          {/* x date labels */}
          {xLabelIdx.map((i) => (
            <text key={i} x={x(dates[i])} y={H - 12} className="pt-axis" textAnchor="middle">
              {dates[i].slice(5)}
            </text>
          ))}

          {/* one polyline per model + endpoint dots */}
          {series.map((s) => {
            const pts = s.points.map((p) => `${x(p.date)},${y(p.value)}`).join(' ');
            const last = s.points[s.points.length - 1];
            return (
              <g key={s.name}>
                <polyline points={pts} fill="none" stroke={s.color} strokeWidth={2} className="pt-line" />
                <circle cx={x(last.date)} cy={y(last.value)} r={3} fill={s.color} />
              </g>
            );
          })}

          {/* legend */}
          {series.map((s, i) => (
            <g key={s.name} transform={`translate(${pad.left + plotW + 14}, ${pad.top + 6 + i * 20})`}>
              <rect width={10} height={10} y={-9} fill={s.color} rx={2} />
              <text x={16} className="pt-legend">{shortModel(s.name)}</text>
            </g>
          ))}
        </svg>
      </div>
    </section>
  );
};

export default PerformanceTrend;
