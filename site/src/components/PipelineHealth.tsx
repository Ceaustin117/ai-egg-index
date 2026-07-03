import React from 'react';
import { RunHealth } from '../types';
import './PipelineHealth.css';

interface PipelineHealthProps {
  data: RunHealth;
}

const shortModel = (m: string) => m.split('/').pop() || m;

/**
 * Compact run-over-run pipeline health from output/run-health.json:
 * a timeline of recent weekly runs (green = all benchmarks scored, amber = one or
 * more errored — usually a provider's free-tier quota), plus the latest run's issues.
 */
const PipelineHealth: React.FC<PipelineHealthProps> = ({ data }) => {
  const runs = data.runs.slice(-12);
  if (runs.length === 0) return null;

  const latest = runs[runs.length - 1];
  const issuesByModel: Record<string, string[]> = {};
  Object.entries(latest.cells).forEach(([model, cells]) => {
    const errored = Object.entries(cells)
      .filter(([, status]) => status === 'error')
      .map(([bench]) => bench);
    if (errored.length) issuesByModel[model] = errored;
  });

  return (
    <section className="pipeline-health">
      <h2>Pipeline Health</h2>
      <p className="ph-sub">
        Each weekly run. Green = every benchmark scored; amber = one or more errored
        (usually a provider's free-tier quota, handled gracefully — not a fake zero).
      </p>

      <div className="ph-timeline">
        {runs.map((run) => {
          const healthy = run.summary.error === 0;
          return (
            <div
              key={run.date}
              className={`ph-run ${healthy ? 'ok' : 'warn'}`}
              title={`${run.date}: ${run.summary.ok}/${run.summary.total} benchmark runs healthy`}
            >
              <span className="ph-dot" aria-hidden="true" />
              <span className="ph-date">{run.date.slice(5)}</span>
            </div>
          );
        })}
      </div>

      <p className="ph-latest">
        Latest run <strong>{latest.date}</strong>: {latest.summary.ok}/{latest.summary.total}{' '}
        benchmark runs healthy
        {Object.keys(issuesByModel).length > 0 && (
          <>
            {' — '}
            <span className="ph-issues">
              {Object.entries(issuesByModel)
                .map(([model, benches]) => `${shortModel(model)} (${benches.join(', ')})`)
                .join('; ')}
            </span>
          </>
        )}
      </p>
    </section>
  );
};

export default PipelineHealth;
