// Post-build: inject a static, crawlable fallback + Dataset JSON-LD into dist/index.html.
// The app is a client-rendered SPA, so without this, AI crawlers (Perplexity/ChatGPT) and
// non-JS bots see an empty page. We render the current rankings as real HTML inside #root
// (React replaces it on mount for JS users) plus a dated quotable sentence + structured
// data — exactly what answer engines cite. Runs after `vite build` (see package.json).
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, '..', 'dist');
const indexPath = join(dist, 'index.html');
const dataPath = join(dist, 'data', 'latest.json');

if (!existsSync(dataPath)) {
  console.warn('[inject-static] no dist/data/latest.json; skipping');
  process.exit(0);
}

const data = JSON.parse(readFileSync(dataPath, 'utf8'));
const BENCH = [
  ['practical_knowledge', 'Practical'],
  ['ifeval', 'IFEval'],
  ['gsm8k', 'GSM8K'],
  ['creative_technical', 'Creative'],
];
const esc = (s) =>
  String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// Overall = the two custom benchmarks every provider runs (IFEval/GSM8K are Groq/Cohere-only).
// null unless the model has BOTH — so it can't rank on a single benchmark.
const OVERALL_KEYS = ['practical_knowledge', 'creative_technical'];
const overall = (b) => {
  const v = OVERALL_KEYS.map((k) => b && b[k]).filter((x) => x && typeof x.score === 'number').map((x) => x.score);
  return v.length === OVERALL_KEYS.length ? v.reduce((a, c) => a + c, 0) / v.length : null;
};
const cell = (x) => (x == null ? '—' : x.status === 'error' || x.score == null ? 'N/A' : Math.round(x.score * 100) + '%');

// Ranked models (both customs) first by overall desc; unranked (null) sorted last.
const models = [...data.models].sort((a, b) => {
  const oa = overall(a.benchmarks), ob = overall(b.benchmarks);
  if (oa === null && ob === null) return 0;
  if (oa === null) return 1;
  if (ob === null) return -1;
  return ob - oa;
});
const top = models.find((m) => overall(m.benchmarks) !== null) || models[0];
const rows = models
  .map((m) => {
    const b = m.benchmarks || {};
    const o = overall(b);
    const tds = BENCH.map(([k]) => `<td>${cell(b[k])}</td>`).join('');
    return `<tr><td>${esc(m.model.split('/').pop())}</td><td>${esc(m.provider || '')}</td>${tds}<td><strong>${o === null ? '—' : Math.round(o * 100) + '%'}</strong></td></tr>`;
  })
  .join('');

const sentence = `As of ${esc(data.last_updated)}, the AI Egg Index ranks free-tier LLMs on everyday tasks. The current top free model overall is ${esc(top.model.split('/').pop())} (${esc(top.provider)}) at ${Math.round(overall(top.benchmarks) * 100)}%. Scores are small-sample and directional.`;

const staticHtml = `
      <main>
        <h1>AI Egg Index — the everyday index for AI</h1>
        <p>${sentence}</p>
        <table>
          <thead><tr><th>Model</th><th>Provider</th><th>Practical</th><th>IFEval</th><th>GSM8K</th><th>Creative</th><th>Overall</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <p>Free-tier only, everyday tasks, updated weekly. Open source:
          <a href="https://github.com/Ceaustin117/ai-egg-index">github.com/Ceaustin117/ai-egg-index</a></p>
      </main>`;

const jsonld = {
  '@context': 'https://schema.org',
  '@type': 'Dataset',
  name: 'AI Egg Index',
  description:
    'Weekly benchmark of free-tier LLMs on everyday tasks (practical knowledge, instruction following, math, coding).',
  url: 'https://ceaustin117.github.io/ai-egg-index/',
  license: 'https://creativecommons.org/licenses/by/4.0/',
  creator: { '@type': 'Person', name: 'Chris Austin', url: 'https://chrisaustin.dev' },
  dateModified: data.last_updated,
  isAccessibleForFree: true,
  distribution: [
    {
      '@type': 'DataDownload',
      encodingFormat: 'application/json',
      contentUrl: 'https://ceaustin117.github.io/ai-egg-index/data/latest.json',
    },
  ],
};

let html = readFileSync(indexPath, 'utf8');
html = html.replace('</head>', `    <script type="application/ld+json">${JSON.stringify(jsonld)}</script>\n  </head>`);
html = html.replace('<div id="root"></div>', `<div id="root">${staticHtml}\n    </div>`);
writeFileSync(indexPath, html);
console.log('[inject-static] injected static fallback table + Dataset JSON-LD');
