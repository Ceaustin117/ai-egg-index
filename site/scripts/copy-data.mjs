// Copy the repo's aggregated output JSON into the site's public/data so the built site
// serves its own benchmark data. Run automatically before dev/build (see package.json).
import { mkdirSync, copyFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const here = dirname(fileURLToPath(import.meta.url));
const outputDir = join(here, '..', '..', 'output'); // <repo>/output
const dataDir = join(here, '..', 'public', 'data'); // <repo>/site/public/data

mkdirSync(dataDir, { recursive: true });

for (const file of ['latest.json', 'historical.json', 'run-health.json']) {
  const src = join(outputDir, file);
  if (existsSync(src)) {
    copyFileSync(src, join(dataDir, file));
    console.log(`[copy-data] ${file}`);
  } else {
    console.warn(`[copy-data] missing ${file} — site will 404 on it`);
  }
}
