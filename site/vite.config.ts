import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Project GitHub Pages site is served from https://ceaustin117.github.io/ai-egg-index/,
// so the base path must match the repo name.
export default defineConfig({
  plugins: [react()],
  base: '/ai-egg-index/',
});
