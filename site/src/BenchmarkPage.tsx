import React, { useState, useEffect } from 'react';
import BenchmarkComparer from './components/BenchmarkComparer';
import PerformanceTrend from './components/PerformanceTrend';
import PipelineHealth from './components/PipelineHealth';
import { BenchmarkData, HistoricalData, RunHealth } from './types';
import { loadBenchmarkFile } from './loadBenchmarkData';
import './BenchmarkPage.css';

type View = 'rankings' | 'about' | 'details' | 'history';

const VIEW_LABELS: Record<View, string> = {
  rankings: 'Rankings',
  about: 'About',
  details: 'Details',
  history: 'History',
};

const BenchmarkPage: React.FC = () => {
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null);
  const [historicalData, setHistoricalData] = useState<HistoricalData | null>(null);
  const [runHealth, setRunHealth] = useState<RunHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // The active view is driven by the `?view=` query param so it can be deep-linked
  // (e.g. from the header dropdown). Falls back to 'rankings' when absent/invalid.
  const initialParam =
    typeof window !== 'undefined'
      ? new URLSearchParams(window.location.search).get('view')
      : null;
  const initialView: View =
    initialParam && initialParam in VIEW_LABELS ? (initialParam as View) : 'rankings';
  const [view, setViewState] = useState<View>(initialView);
  const setView = (v: View) => {
    setViewState(v);
    const params = new URLSearchParams(window.location.search);
    if (v === 'rankings') params.delete('view');
    else params.set('view', v);
    const qs = params.toString();
    window.history.replaceState(
      null,
      '',
      qs ? `?${qs}${window.location.hash}` : window.location.pathname,
    );
  };

  useEffect(() => {
    const loadData = async () => {
      try {
        const [latestData, histData, healthData] = await Promise.all([
          loadBenchmarkFile<BenchmarkData>('latest.json'),
          loadBenchmarkFile<HistoricalData>('historical.json').catch(() => null),
          loadBenchmarkFile<RunHealth>('run-health.json').catch(() => null)
        ]);

        setBenchmarkData(latestData);
        if (histData) setHistoricalData(histData);
        if (healthData) setRunHealth(healthData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  return (
    <>
      <div className="benchmark-page">
        <div className="benchmark-page-container">
          <a href="https://chrisaustin.dev" className="back-link">← chrisaustin.dev</a>

          <header className="benchmark-header">
            <h1>AI Egg Index</h1>
            <p className="benchmark-tagline">
              Benchmarking free tier LLMs on tasks regular people actually care about
            </p>

            <nav className="benchmark-tabs" role="tablist">
              {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
                <button
                  key={v}
                  type="button"
                  role="tab"
                  aria-selected={v === view}
                  className={`benchmark-tab ${v === view ? 'active' : ''}`}
                  onClick={() => setView(v)}
                >
                  {VIEW_LABELS[v]}
                </button>
              ))}
            </nav>
          </header>

          {view === 'rankings' && (
          <section className="benchmark-results">
            <h2>Current Rankings</h2>
            {loading ? (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading benchmark data...</p>
              </div>
            ) : error ? (
              <div className="error-state">
                <p>Error: {error}</p>
                <p className="error-hint">Benchmark data may not be deployed yet.</p>
              </div>
            ) : benchmarkData ? (
              <>
                <p className="last-updated">
                  Runs every week • Last updated: {benchmarkData.last_updated}
                </p>
                <p className="sample-caveat">
                  Small-sample scores — read as a directional signal, not a precise ranking.
                </p>
                <BenchmarkComparer
                  data={benchmarkData.models}
                  historicalData={historicalData || undefined}
                />
              </>
            ) : null}
          </section>
          )}

          {view === 'about' && (
          <>
          <section className="benchmark-intro">
            <div className="intro-content">
              <h2>About the AI Egg Index</h2>
              <p>
                Why "Egg Index"? Most AI benchmarks are like complex economic models—abstract
                scores that don't mean much to regular people. This aims to be more like the egg
                price index: a simple, grounded measure of what actually matters. Can free AI help
                with your taxes? Can it handle a complex reasoning task that someone might actually ask?
              </p>
            </div>
          </section>

          <section className="benchmark-intro">
            <div className="intro-content">
              <h2>Why This Matters</h2>
              <p>
                Academic benchmarks tell researchers how models perform on standardized tests.
                But what about the rest of us? Can free AI actually help with taxes?
                Will it follow complex instructions? Can it solve real math problems?
              </p>
              <p>
                The AI Egg Index tracks free tier LLM performance over time on tasks
                that matter to everyday users—like tracking egg prices tracks the economy.
              </p>
            </div>
          </section>

          <section className="benchmark-landscape">
            <h2>How We're Different</h2>
            <p className="landscape-intro">
              Great AI benchmarks already exist, but they serve different purposes. Here's where we fit in:
            </p>
            <div className="comparison-table-wrapper">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Benchmark</th>
                    <th>Focus</th>
                    <th>Free Tier?</th>
                    <th>Practical Tasks?</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard" target="_blank" rel="noopener noreferrer">
                        LMSYS Chatbot Arena
                      </a>
                    </td>
                    <td>Human preference voting</td>
                    <td className="cell-no">All models</td>
                    <td className="cell-partial">Vibes, not tasks</td>
                  </tr>
                  <tr>
                    <td>
                      <a href="https://www.vals.ai/benchmarks/ace" target="_blank" rel="noopener noreferrer">
                        ACE (AI Consumer Index)
                      </a>
                    </td>
                    <td>Consumer activities</td>
                    <td className="cell-no">Frontier models</td>
                    <td className="cell-yes">Yes</td>
                  </tr>
                  <tr>
                    <td>
                      <a href="https://scale.com/leaderboard" target="_blank" rel="noopener noreferrer">
                        Scale Leaderboard
                      </a>
                    </td>
                    <td>Enterprise tool use</td>
                    <td className="cell-no">Enterprise</td>
                    <td className="cell-no">Dev/Agent tasks</td>
                  </tr>
                  <tr>
                    <td>
                      <a href="https://github.com/openai/simple-evals" target="_blank" rel="noopener noreferrer">
                        OpenAI Evals
                      </a>
                    </td>
                    <td>Academic benchmarks</td>
                    <td className="cell-no">All models</td>
                    <td className="cell-no">Academic</td>
                  </tr>
                  <tr className="highlight-row">
                    <td><strong>AI Egg Index</strong></td>
                    <td>Everyday layperson tasks</td>
                    <td className="cell-yes">Free only</td>
                    <td className="cell-yes">Yes</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="differentiators">
              <div className="diff-card">
                <span className="diff-icon">🆓</span>
                <h4>Free Tier Reality</h4>
                <p>We test what you can actually use without paying—including rate limits and context constraints.</p>
              </div>
              <div className="diff-card">
                <span className="diff-icon">✅</span>
                <h4>Task Completion</h4>
                <p>Not "which response feels better" but "can it actually answer my tax question correctly?"</p>
              </div>
              <div className="diff-card">
                <span className="diff-icon">🔄</span>
                <h4>Reproducible</h4>
                <p>Versioned prompts, fixed test sets, public rubrics. Anyone can verify our results.</p>
              </div>
            </div>
          </section>
          </>
          )}

          {view === 'details' && (
          <>
          <section className="benchmark-categories">
            <h2>The 4 Benchmarks</h2>
            <div className="category-grid">
              <div className="category-card custom">
                <div className="category-badge">Custom</div>
                <h3>Practical Knowledge</h3>
                <p>
                  Real-world info people actually search for: tax brackets, retirement limits,
                  minimum wage, consumer rights. Tests if models have current, accurate knowledge.
                </p>
              </div>
              <div className="category-card openbench">
                <div className="category-badge">OpenBench</div>
                <h3>IFEval</h3>
                <p>
                  Instruction following. When you say "format as a table" or "respond in exactly
                  3 bullet points"—does the model actually do it?
                </p>
              </div>
              <div className="category-card openbench">
                <div className="category-badge">OpenBench</div>
                <h3>GSM8K</h3>
                <p>
                  Grade school math word problems. If the model can't help your kid with homework,
                  is it really that useful?
                </p>
              </div>
              <div className="category-card custom">
                <div className="category-badge">Custom</div>
                <h3>Creative+Technical</h3>
                <p>
                  Push the limits: "Write a budget tracker that outputs summaries as haiku."
                  Tests whether models can combine creativity with technical execution.
                </p>
              </div>
            </div>
          </section>

          <section className="benchmark-methodology">
            <h2>Methodology</h2>
            <div className="methodology-grid">
              <div className="methodology-card">
                <h3>Free Tier Only</h3>
                <p>
                  We only test models available for free. If you have to pay, it's not included.
                  This keeps the index practical for regular users.
                </p>
              </div>
              <div className="methodology-card">
                <h3>LLM-as-Judge</h3>
                <p>
                  Custom benchmarks use LLM-as-judge scoring with rubrics for factual accuracy,
                  completeness, and recency awareness. The judge is a stronger free-tier model
                  (Llama 3.3 70B) — deliberately larger than the models under test, since the
                  grader is infrastructure, not a contestant.
                </p>
              </div>
              <div className="methodology-card">
                <h3>Directional, Not Precise</h3>
                <p>
                  Each benchmark runs on a small sample per cycle, so scores are best read as a
                  rough, directional signal — not a precise ranking. Expect some week-to-week
                  movement to be sampling noise rather than real change.
                </p>
              </div>
              <div className="methodology-card">
                <h3>Historical Tracking</h3>
                <p>
                  We run the full benchmark suite once a week (automated every Sunday) and track
                  changes over time. The sparklines in the table show recent runs — read them for
                  broad direction, not exact deltas (see "Directional, Not Precise").
                </p>
              </div>
              <div className="methodology-card">
                <h3>Open Source</h3>
                <p>
                  All benchmark definitions, evaluation scripts, and results are open source.
                  Check the repo to see exactly how we score.
                </p>
                <a
                  href="https://github.com/Ceaustin117/ai-egg-index"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="github-repo-link"
                >
                  View on GitHub
                </a>
              </div>
            </div>
          </section>

          <section className="benchmark-providers">
            <h2>Free Tier Providers Tested</h2>
            <div className="provider-list">
              <div className="provider-item">
                <span className="provider-name groq">Groq</span>
                <span className="provider-limit">30 req/min</span>
              </div>
              <div className="provider-item">
                <span className="provider-name google">Google</span>
                <span className="provider-limit">10 req/min</span>
              </div>
              <div className="provider-item">
                <span className="provider-name cohere">Cohere</span>
                <span className="provider-limit">100 req/min</span>
              </div>
              <div className="provider-item">
                <span className="provider-name huggingface">HuggingFace</span>
                <span className="provider-limit">Rate limited</span>
              </div>
            </div>
          </section>
          </>
          )}

          {view === 'history' && (
          <section className="benchmark-history">
            <h2>History</h2>
            <p className="history-intro">
              How models have performed over time, and the health of each weekly run.
            </p>
            {historicalData && <PerformanceTrend data={historicalData} />}
            {runHealth && <PipelineHealth data={runHealth} />}
            {!historicalData && !runHealth && (
              <p className="error-hint">History data is not available yet.</p>
            )}
          </section>
          )}
        </div>
      </div>
      <footer className="site-footer">
        <p>
          Open source ·{' '}
          <a href="https://github.com/Ceaustin117/ai-egg-index">github.com/Ceaustin117/ai-egg-index</a>{' '}
          · by <a href="https://chrisaustin.dev">Chris Austin</a>
        </p>
      </footer>
    </>
  );
};

export default BenchmarkPage;
