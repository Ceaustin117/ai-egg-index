import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages serves from https://ceaustin117.github.io/ai-egg-index/ (base must match
// the repo name). A Hugging Face static Space serves from root, so build it with
// SITE_BASE=/ (see docs/HUGGINGFACE.md).
export default defineConfig({
  plugins: [react()],
  base: process.env.SITE_BASE || '/ai-egg-index/',
});
