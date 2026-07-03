import React, { useState, useMemo } from 'react';
import { BenchmarkResult, HistoricalData } from '../types';
import './BenchmarkComparer.css';

interface BenchmarkComparerProps {
  data: BenchmarkResult[];
  historicalData?: HistoricalData;
}

type SortField = 'model' | 'provider' | 'practical_knowledge' | 'ifeval' | 'gsm8k' | 'creative_technical' | 'overall';
type SortDirection = 'asc' | 'desc';

const BENCHMARK_INFO: Record<string, { name: string; description: string; category: string }> = {
  practical_knowledge: {
    name: 'Practical Knowledge',
    description: 'Real-world info: taxes, regulations, finance',
    category: 'Custom'
  },
  ifeval: {
    name: 'IFEval',
    description: 'Following complex instructions',
    category: 'OpenBench'
  },
  gsm8k: {
    name: 'GSM8K',
    description: 'Math word problems',
    category: 'OpenBench'
  },
  creative_technical: {
    name: 'Creative+Technical',
    description: 'Coding with creative constraints',
    category: 'Custom'
  }
};

const BENCHMARK_KEYS = ['practical_knowledge', 'ifeval', 'gsm8k', 'creative_technical'] as const;

const BenchmarkComparer: React.FC<BenchmarkComparerProps> = ({ data, historicalData }) => {
  const [sortField, setSortField] = useState<SortField>('overall');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null);
  const [providerFilter, setProviderFilter] = useState<string>('all');

  const providers = useMemo(() => {
    const uniqueProviders = Array.from(new Set(data.map(d => d.provider)));
    return ['all', ...uniqueProviders.sort()];
  }, [data]);

  const calculateOverall = (model: BenchmarkResult): number => {
    const scores = BENCHMARK_KEYS.map(key => model.benchmarks[key]?.score ?? 0);
    const validScores = scores.filter(s => s > 0);
    return validScores.length > 0 ? validScores.reduce((a, b) => a + b, 0) / validScores.length : 0;
  };

  const getScoreClass = (score: number): string => {
    if (score >= 0.75) return 'score-excellent';
    if (score >= 0.6) return 'score-good';
    if (score >= 0.4) return 'score-fair';
    return 'score-poor';
  };

  const formatScore = (score: number | null | undefined): string => {
    if (score === undefined || score === null || score === 0) return '—';
    return (score * 100).toFixed(0) + '%';
  };

  // A benchmark that was attempted but failed (provider unavailable) has score === null
  // with status 'error'. Distinct from a benchmark that simply wasn't run (undefined).
  const isBenchError = (bench?: BenchmarkResult['benchmarks'][keyof BenchmarkResult['benchmarks']]): boolean =>
    !!bench && (bench.status === 'error' || bench.score === null);

  // A whole row is "unavailable" when every benchmark either failed or was never run.
  const isModelUnavailable = (model: BenchmarkResult): boolean =>
    BENCHMARK_KEYS.every(key => {
      const bench = model.benchmarks[key];
      return !bench || isBenchError(bench);
    });

  const getTrend = (model: string, benchmark: string): 'up' | 'down' | 'neutral' | null => {
    if (!historicalData?.models[model]) return null;
    // Prefer the smoothed rolling series (noise-resistant); fall back to raw scores.
    const scores = historicalData.models[model].rolling ?? historicalData.models[model].scores;
    if (scores.length < 2) return null;

    const latest = scores[scores.length - 1][benchmark as keyof typeof scores[0]] as number | undefined;
    const previous = scores[scores.length - 2][benchmark as keyof typeof scores[0]] as number | undefined;

    if (latest === undefined || previous === undefined) return null;
    if (latest > previous + 0.02) return 'up';
    if (latest < previous - 0.02) return 'down';
    return 'neutral';
  };

  const renderTrendIcon = (trend: 'up' | 'down' | 'neutral' | null) => {
    if (!trend) return null;
    if (trend === 'up') return <span className="trend-icon trend-up">↑</span>;
    if (trend === 'down') return <span className="trend-icon trend-down">↓</span>;
    return <span className="trend-icon trend-neutral">→</span>;
  };

  const sortedData = useMemo(() => {
    let filtered = providerFilter === 'all' ? data : data.filter(d => d.provider === providerFilter);

    return [...filtered].sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      if (sortField === 'model' || sortField === 'provider') {
        aVal = a[sortField].toLowerCase();
        bVal = b[sortField].toLowerCase();
      } else if (sortField === 'overall') {
        aVal = calculateOverall(a);
        bVal = calculateOverall(b);
      } else {
        aVal = a.benchmarks[sortField]?.score ?? 0;
        bVal = b.benchmarks[sortField]?.score ?? 0;
      }

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDirection === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });
  }, [data, sortField, sortDirection, providerFilter]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const renderSortIndicator = (field: SortField) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? ' ↑' : ' ↓';
  };

  const renderSparkline = (model: string, benchmark: string) => {
    if (!historicalData?.models[model]) return null;
    // Prefer the smoothed rolling series (noise-resistant); fall back to raw scores.
    const scores = historicalData.models[model].rolling ?? historicalData.models[model].scores;
    if (scores.length < 2) return null;

    const values = scores.map(s => (s[benchmark as keyof typeof s] as number) ?? 0).filter(v => v > 0);
    if (values.length < 2) return null;

    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;
    const width = 60;
    const height = 20;
    const padding = 2;

    const points = values.map((v, i) => {
      const x = padding + (i / (values.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((v - min) / range) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="sparkline" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <polyline
          fill="none"
          stroke="#06b6d4"
          strokeWidth="1.5"
          points={points}
        />
      </svg>
    );
  };

  return (
    <div className="benchmark-comparer">
      <div className="benchmark-controls">
        <div className="filter-group">
          <label htmlFor="provider-filter">Provider:</label>
          <select
            id="provider-filter"
            value={providerFilter}
            onChange={(e) => setProviderFilter(e.target.value)}
          >
            {providers.map(p => (
              <option key={p} value={p}>
                {p === 'all' ? 'All Providers' : p.charAt(0).toUpperCase() + p.slice(1)}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="benchmark-table-wrapper">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('model')} className="sortable">
                Model{renderSortIndicator('model')}
              </th>
              <th onClick={() => handleSort('provider')} className="sortable">
                Provider{renderSortIndicator('provider')}
              </th>
              {BENCHMARK_KEYS.map(key => (
                <th
                  key={key}
                  onClick={() => handleSort(key as SortField)}
                  onMouseEnter={() => setSelectedBenchmark(key)}
                  onMouseLeave={() => setSelectedBenchmark(null)}
                  className="sortable benchmark-header"
                  title={BENCHMARK_INFO[key].description}
                >
                  <span className="benchmark-name">{BENCHMARK_INFO[key].name}</span>
                  <span className="benchmark-category">{BENCHMARK_INFO[key].category}</span>
                  {renderSortIndicator(key as SortField)}
                </th>
              ))}
              <th onClick={() => handleSort('overall')} className="sortable overall-header">
                Overall{renderSortIndicator('overall')}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedData.map((model) => {
              const overall = calculateOverall(model);
              const unavailable = isModelUnavailable(model);
              return (
                <tr key={model.model} className={unavailable ? 'row-unavailable' : ''}>
                  <td className="model-name">
                    <span className="model-text">{model.model.split('/').pop()}</span>
                    {model.metadata?.context_window && (
                      <span className="model-meta">{(model.metadata.context_window / 1000).toFixed(0)}K ctx</span>
                    )}
                    {unavailable && <span className="unavailable-badge">unavailable</span>}
                  </td>
                  <td className="provider-cell">
                    <span className={`provider-badge provider-${model.provider}`}>
                      {model.provider}
                    </span>
                  </td>
                  {BENCHMARK_KEYS.map(key => {
                    const bench = model.benchmarks[key];
                    const score = bench?.score;
                    const errored = isBenchError(bench);
                    const trend = getTrend(model.model, key);
                    return (
                      <td
                        key={key}
                        className={`score-cell ${errored ? 'score-unavailable' : (score ? getScoreClass(score) : '')} ${selectedBenchmark === key ? 'highlighted' : ''}`}
                        title={errored ? `Unavailable this run: ${bench?.error ?? 'provider error'}` : undefined}
                      >
                        <div className="score-content">
                          {errored ? (
                            <span className="score-value score-na">N/A</span>
                          ) : (
                            <>
                              <span className="score-value">{formatScore(score)}</span>
                              {renderTrendIcon(trend)}
                              {renderSparkline(model.model, key)}
                            </>
                          )}
                        </div>
                      </td>
                    );
                  })}
                  <td className={`score-cell overall-cell ${unavailable ? 'score-unavailable' : getScoreClass(overall)}`}>
                    <span className="score-value overall-value">{unavailable ? 'N/A' : formatScore(overall)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {selectedBenchmark && (
        <div className="benchmark-tooltip">
          <strong>{BENCHMARK_INFO[selectedBenchmark].name}</strong>
          <p>{BENCHMARK_INFO[selectedBenchmark].description}</p>
        </div>
      )}

      <div className="benchmark-legend">
        <div className="legend-item">
          <span className="legend-color score-excellent"></span>
          <span>Excellent (75%+)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color score-good"></span>
          <span>Good (60-74%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color score-fair"></span>
          <span>Fair (40-59%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color score-poor"></span>
          <span>Poor (&lt;40%)</span>
        </div>
      </div>
    </div>
  );
};

export default BenchmarkComparer;
