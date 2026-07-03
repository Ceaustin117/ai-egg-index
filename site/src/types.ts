export interface Project {
  id: string;
  title: string;
  description: string;
  technologies: string[];
  imageUrl?: string;
  liveUrl?: string;
  githubUrl?: string;
  featured: boolean;
}

export interface Skill {
  name: string;
  category: 'ai' | 'ml' | 'cloud' | 'tools' | 'all';
  proficiency: 'beginner' | 'intermediate' | 'advanced' | 'expert';
}

export interface Experience {
  id: string;
  company: string;
  position: string;
  startDate: string;
  endDate?: string;
  description: string;
  technologies: string[];
}

export interface ContactInfo {
  email: string;
  linkedin?: string;
  github?: string;
  twitter?: string;
  phone?: string;
}

export interface PersonalInfo {
  name: string;
  title: string;
  summary: string;
  location: string;
  contact: ContactInfo;
}

export interface NavSubItem {
  id: string;
  label: string;
}

export interface NavItem {
  id: string;
  label: string;
  href: string;
  // When present, this nav item renders as a dropdown instead of a plain link.
  subItems?: NavSubItem[];
}

// Benchmark types for the Practical AI Index
export interface BenchmarkScoreDetails {
  [key: string]: number;
}

export interface BenchmarkScore {
  // null when the benchmark could not be run (e.g. the provider's free tier was
  // unavailable that run). Distinguishes "we tried and the API failed" from a real low score.
  score: number | null;
  status?: 'ok' | 'error';
  error?: string;
  details?: BenchmarkScoreDetails;
  pass_at_1?: number;
  strict?: number;
  loose?: number;
  accuracy?: number;
}

export interface BenchmarkResult {
  model: string;
  provider: string;
  tier: 'free' | 'paid';
  date: string;
  benchmarks: {
    practical_knowledge?: BenchmarkScore;
    ifeval?: BenchmarkScore;
    gsm8k?: BenchmarkScore;
    creative_technical?: BenchmarkScore;
  };
  metadata?: {
    knowledge_cutoff?: string;
    context_window?: number;
    rate_limit?: string;
  };
}

export interface BenchmarkData {
  last_updated: string;
  models: BenchmarkResult[];
}

export interface HistoricalScoreEntry {
  date: string;
  practical_knowledge?: number;
  ifeval?: number;
  gsm8k?: number;
  creative_technical?: number;
}

export interface HistoricalModelData {
  provider: string;
  tier: string;
  scores: HistoricalScoreEntry[];
  // Trailing-window smoothed series (same shape as scores). Present when the
  // aggregator emits it; prefer it over raw `scores` for noise-resistant trends.
  rolling?: HistoricalScoreEntry[];
}

export interface HistoricalData {
  last_updated: string;
  models: {
    [modelName: string]: HistoricalModelData;
  };
}

// Run-health tracking (output/run-health.json) — per-run pipeline health.
export type CellStatus = 'ok' | 'error';

export interface RunHealthEntry {
  date: string;
  models: string[];
  cells: {
    [modelName: string]: { [benchmark: string]: CellStatus };
  };
  summary: { ok: number; error: number; total: number };
}

export interface RunHealth {
  last_updated: string;
  runs: RunHealthEntry[];
}