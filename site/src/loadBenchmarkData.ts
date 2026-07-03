// The project site bundles its own benchmark data (copied from the repo's output/ at
// build time into public/data/), so it loads from its own origin — no dependency on
// raw.githubusercontent.com. import.meta.env.BASE_URL is "/ai-egg-index/" in production.
type BenchmarkFile = 'latest.json' | 'historical.json' | 'run-health.json';

export async function loadBenchmarkFile<T>(file: BenchmarkFile): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${file}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to load ${file} (HTTP ${res.status})`);
  }
  return (await res.json()) as T;
}
