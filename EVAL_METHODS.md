# LLM Evaluation Methods — Ranked by Citation Count

A cross-referenced inventory of evaluation methods extracted from 20 authoritative sources (academic papers, lab guides, practitioner blogs, and framework docs). Methods are ranked by **number of independent sources that reference them**.

---

## Sources Key

| ID | Source | Type |
|----|--------|------|
| S1 | [BetterBench (Stanford, NeurIPS 2024)](https://arxiv.org/html/2411.12990v1) | Academic paper |
| S2 | [A Survey on LLM-as-a-Judge](https://arxiv.org/html/2411.15594v6) | Academic survey |
| S3 | [LLMs-as-Judges: Comprehensive Survey](https://arxiv.org/html/2412.05579v2) | Academic survey |
| S4 | [Anthropic — Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Lab guide |
| S5 | [Anthropic — Statistical Approach to Model Evals](https://www.anthropic.com/research/statistical-approach-to-model-evals) | Lab research |
| S6 | [OpenAI — Evaluation Best Practices](https://platform.openai.com/docs/guides/evaluation-best-practices) | Lab guide |
| S7 | [Google — Stax Evaluation Best Practices](https://developers.google.com/stax/best-practices) | Lab guide |
| S8 | [Google DeepMind — FACTS Benchmark](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/) | Lab research |
| S9 | [Microsoft — How to Evaluate LLMs](https://www.microsoft.com/en-us/research/articles/how-to-evaluate-llms-a-complete-metric-framework/) | Lab guide |
| S10 | [Hamel Husain — Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) | Practitioner |
| S11 | [Hamel Husain — Using LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/) | Practitioner |
| S12 | [Eugene Yan — Task-Specific LLM Evals](https://eugeneyan.com/writing/evals/) | Practitioner |
| S13 | [Eugene Yan — LLM Evaluators Effectiveness](https://eugeneyan.com/writing/llm-evaluators/) | Practitioner |
| S14 | [Eugene Yan — Patterns for LLM Systems](https://eugeneyan.com/writing/llm-patterns/) | Practitioner |
| S15 | [HuggingFace — LLM-as-a-Judge Cookbook](https://huggingface.co/learn/cookbook/en/llm_judge) | Framework guide |
| S16 | [Evidently AI — LLM-as-a-Judge Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) | Framework guide |
| S17 | [RAGAS Documentation](https://docs.ragas.io/en/latest/concepts/metrics/) | Framework docs |
| S18 | [Braintrust — LLM Evaluation Metrics Guide](https://www.braintrust.dev/articles/llm-evaluation-metrics-guide) | Framework guide |
| S19 | [Vals.ai — Methodology](https://www.vals.ai/methodology) | Benchmark org |
| S20 | [Confident AI — LLM-as-a-Judge Guide](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method) | Framework guide |
| S21 | [Inspect AI (UK AISI)](https://inspect.aisi.org.uk/) | Framework docs |

---

## Tier 1: Universal Methods (10+ sources)

### 1. LLM-as-a-Judge / Model-Graded Evaluation (18 sources)
Using an LLM to score or judge the output of another LLM based on criteria defined in a prompt.

**Also known as:** Auto-rater, model-based grader, LLM evaluator, autorater

| Sources |
|---------|
| S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S20 |

---

### 2. Human Evaluation / Human Grading (15 sources)
Manual review and labeling of LLM outputs by human annotators, used as ground truth calibration.

**Also known as:** Human annotation, manual labeling, human raters, expert review

| Sources |
|---------|
| S1, S2, S3, S4, S6, S7, S9, S10, S11, S12, S13, S15, S16, S18, S20 |

---

### 3. Rubric-Based Scoring (14 sources)
Evaluating outputs against a structured rubric that defines what each score level means.

**Also known as:** Criteria-based evaluation, scoring guidelines, grading scale, Likert scale

| Sources |
|---------|
| S1, S2, S3, S4, S6, S11, S13, S15, S16, S17, S18, S19, S20, S21 |

---

### 4. Pairwise Comparison (13 sources)
Comparing two outputs side by side and selecting the better one (win/tie/loss).

**Also known as:** A/B comparison, head-to-head evaluation, preference judgment

| Sources |
|---------|
| S2, S3, S4, S6, S12, S13, S14, S15, S16, S18, S19, S20, S21 |

---

### 5. Chain-of-Thought (CoT) Prompting for Judges (12 sources)
Instructing the judge LLM to explain its reasoning step-by-step before giving a final score.

**Also known as:** Step-by-step reasoning, evaluation rationale, explanation-based evaluation

| Sources |
|---------|
| S2, S3, S4, S6, S11, S13, S14, S15, S16, S18, S20, S21 |

---

### 6. Few-Shot Prompting for Judges (11 sources)
Including example evaluations in the judge prompt to calibrate scoring behavior.

**Also known as:** In-context learning for evaluation, example-guided judging

| Sources |
|---------|
| S2, S3, S6, S11, S13, S14, S15, S16, S18, S20, S21 |

---

### 7. Code-Based / Rule-Based Evaluation (11 sources)
Deterministic checks using exact match, regex, string matching, or programmatic assertions.

**Also known as:** Heuristic checks, assertion-based evaluation, unit tests for LLMs

| Sources |
|---------|
| S1, S4, S7, S9, S10, S12, S16, S17, S18, S19, S20 |

---

### 8. Binary Pass/Fail Scoring (10 sources)
Simple yes/no or pass/fail judgments rather than multi-point scales.

**Also known as:** Binary classification, yes/no evaluation, thumbs up/down

| Sources |
|---------|
| S2, S3, S4, S10, S11, S12, S16, S17, S18, S20 |

---

### 9. Reference-Based Evaluation (10 sources)
Comparing generated output against a known-correct gold-standard reference answer.

**Also known as:** Gold-standard comparison, ground truth matching, expected output comparison

| Sources |
|---------|
| S1, S3, S6, S12, S13, S14, S16, S17, S18, S20 |

---

### 10. Faithfulness / Groundedness (10 sources)
Checking whether the response is grounded in provided source material without hallucination.

**Also known as:** Hallucination detection, factual grounding, context fidelity

| Sources |
|---------|
| S3, S8, S9, S12, S14, S16, S17, S18, S19, S20 |

---

## Tier 2: Widely Adopted Methods (5–9 sources)

### 11. G-Eval (9 sources)
Framework using CoT prompting + form-filling + token probability normalization for scoring.

| Sources |
|---------|
| S2, S3, S13, S14, S15, S16, S18, S20, S21 |

---

### 12. Position Bias Mitigation / Position Swapping (9 sources)
Shuffling or swapping the order of responses in pairwise comparisons to detect and reduce position bias.

**Also known as:** Order swapping, content shuffling, debiasing

| Sources |
|---------|
| S2, S3, S4, S6, S11, S13, S15, S16, S20 |

---

### 13. Human-Judge Agreement Measurement (9 sources)
Measuring correlation between LLM judge scores and human annotations (Pearson, Cohen's Kappa, etc.).

**Also known as:** Inter-rater reliability, judge validation, calibration against humans

| Sources |
|---------|
| S2, S3, S5, S10, S11, S13, S15, S16, S20 |

---

### 14. Fine-Tuned Evaluator Models (8 sources)
Training specialized judge models on evaluation data (e.g., Prometheus, JudgeLM, PandaLM, Auto-J).

**Also known as:** Custom judge models, trained evaluators, task-specific judges

| Sources |
|---------|
| S2, S3, S4, S13, S14, S16, S18, S20 |

---

### 15. A/B Testing (8 sources)
Controlled experiments comparing two system variants with real users.

**Also known as:** Online experimentation, controlled comparison, split testing

| Sources |
|---------|
| S4, S7, S9, S10, S12, S16, S18, S19 |

---

### 16. Exact Match (7 sources)
Binary check of whether output exactly matches the expected answer string.

| Sources |
|---------|
| S1, S3, S6, S12, S17, S18, S21 |

---

### 17. Semantic Similarity (7 sources)
Comparing outputs using embedding models (cosine similarity, BERTScore, etc.).

**Also known as:** Embedding-based comparison, BERTScore, MoverScore

| Sources |
|---------|
| S3, S12, S14, S16, S17, S18, S20 |

---

### 18. Pointwise / Direct Scoring (7 sources)
Assessing a single output independently on a numeric scale without comparison.

**Also known as:** Single-output scoring, absolute scoring, Likert scoring

| Sources |
|---------|
| S2, S3, S4, S13, S15, S16, S20 |

---

### 19. BLEU / ROUGE / Traditional NLP Metrics (7 sources)
N-gram overlap metrics comparing generated text to reference text.

| Sources |
|---------|
| S3, S12, S14, S16, S17, S18, S20 |

---

### 20. Regression Testing (7 sources)
Ensuring new changes don't degrade performance on previously passing test cases.

**Also known as:** Non-regression evals, golden set testing

| Sources |
|---------|
| S1, S4, S6, S7, S10, S18, S19 |

---

### 21. Multi-Judge / Ensemble Evaluation (7 sources)
Using multiple LLM judges and aggregating their scores (voting, averaging, consensus).

**Also known as:** Panel of LLM judges (PoLL), multi-LLM consensus, ensemble aggregation

| Sources |
|---------|
| S3, S4, S8, S13, S15, S16, S20 |

---

### 22. Error Analysis / Failure Categorization (6 sources)
Systematically categorizing errors and failure modes to identify improvement priorities.

**Also known as:** Root cause analysis, failure taxonomy, error classification

| Sources |
|---------|
| S4, S10, S11, S12, S18, S19 |

---

### 23. Context Precision / Context Recall (RAG-specific) (6 sources)
Measuring whether retrieved context is relevant (precision) and complete (recall).

| Sources |
|---------|
| S3, S9, S12, S17, S18, S19 |

---

### 24. Statistical Significance Testing (6 sources)
Computing confidence intervals, p-values, or paired-differences tests to validate score differences.

**Also known as:** Paired-differences test, confidence intervals, SEM

| Sources |
|---------|
| S1, S2, S5, S9, S15, S19 |

---

### 25. Listwise / Ranking Evaluation (5 sources)
Ranking multiple outputs collectively rather than comparing pairs.

| Sources |
|---------|
| S2, S3, S6, S13, S21 |

---

### 26. Synthetic Data Generation for Evals (5 sources)
Using LLMs to generate test cases covering diverse scenarios and edge cases.

| Sources |
|---------|
| S3, S10, S11, S12, S18 |

---

### 27. Partial Credit Scoring (5 sources)
Grading task components separately rather than all-or-nothing.

**Also known as:** Additive scoring, component-based grading, decomposed evaluation

| Sources |
|---------|
| S4, S6, S15, S17, S21 |

---

### 28. Data Contamination Detection (5 sources)
Identifying whether benchmark test data leaked into model training sets.

**Also known as:** Canary strings, encrypted test sets, post-cutoff testing

| Sources |
|---------|
| S1, S2, S3, S8, S14 |

---

## Tier 3: Specialized Methods (3–4 sources)

### 29. Criteria Decomposition (4 sources)
Breaking evaluation into multiple independent dimensions scored separately.

| Sources: S2, S3, S15, S17 |

### 30. Question-Answer Generation (QAG) (4 sources)
Breaking evaluations into atomic yes/no questions about specific claims.

| Sources: S3, S12, S17, S20 |

### 31. Token Probability / Logit-Based Scoring (4 sources)
Using raw output token probabilities instead of generated text for scoring.

| Sources: S2, S3, S5, S20 |

### 32. Self-Consistency / Self-Check (4 sources)
Generating multiple samples and measuring agreement to detect hallucinations.

| Sources: S3, S13, S14, S17 |

### 33. Constrained / Structured Output for Judges (4 sources)
Enforcing JSON, regex, or other structured formats on judge output for reliable parsing.

| Sources: S2, S13, S15, S18 |

### 34. Reward Modeling / RLHF-Based Evaluation (4 sources)
Training reward models on human preferences to score outputs.

| Sources: S2, S3, S13, S14 |

### 35. Eval-Driven Development (4 sources)
Writing evaluation tasks before building the system, then iterating until performance improves.

| Sources: S4, S6, S7, S10 |

### 36. Adversarial / Robustness Testing (4 sources)
Testing judges and models against manipulated or adversarial inputs.

| Sources: S1, S2, S3, S19 |

### 37. Latency / Cost Measurement (4 sources)
Tracking response time and token costs as evaluation dimensions.

| Sources: S7, S9, S18, S19 |

### 38. NLI-Based Factual Consistency (3 sources)
Using natural language inference models to detect contradictions between source and output.

| Sources: S3, S12, S17 |

### 39. Human-AI Collaborative Evaluation (3 sources)
Integrating human judgment with LLM evaluation in a hybrid workflow (COEVAL, EvalGen).

| Sources: S3, S4, S10 |

### 40. Domain-Specific / Specialized Judges (3 sources)
Training or prompting judges specifically for a domain (medical, legal, code, etc.).

| Sources: S3, S11, S19 |

### 41. Multi-Turn / Conversation-Level Evaluation (3 sources)
Evaluating entire multi-turn dialogues rather than single responses.

| Sources: S4, S16, S21 |

### 42. Benchmark Versioning / Dynamic Benchmarks (3 sources)
Periodically updating test sets to prevent saturation and contamination.

| Sources: S1, S3, S14 |

### 43. Coverage / Completeness Checks (3 sources)
Verifying that responses include all required key facts or components.

| Sources: S4, S17, S18 |

### 44. Pearson / Spearman / Kendall Correlation (3 sources)
Statistical correlation metrics for comparing judge rankings to ground truth.

| Sources: S3, S5, S15 |

### 45. Transcript / Trace Review (3 sources)
Manual inspection of full interaction logs to validate system behavior.

| Sources: S4, S10, S11 |

---

## Tier 4: Emerging / Niche Methods (1–2 sources)

| Method | Description | Sources |
|--------|-------------|---------|
| Best-of-N Sampling | Generate N outputs, use judge to pick the best | S3 |
| Tree/Graph of Thought | Explore solution spaces with judge-evaluated branches | S3 |
| Meta-Rewarding | Judges themselves optimized through feedback loops | S3 |
| RLAIF | Reinforcement learning from AI feedback | S3 |
| Self-Taught Evaluator | Iterative training using synthetic contrasting outputs | S2 |
| Shadow Experiments | Run treatment/control simultaneously, show only control | S7, S9 |
| Edit Distance / Reformulation Tracking | Measure user prompt rewrites as dissatisfaction signal | S9 |
| Retention Metrics | Track user engagement over time as quality proxy | S7, S9 |
| COMETKiwi | Reference-free translation quality assessment | S12 |
| HyDE | Generate hypothetical docs to improve retrieval evaluation | S14 |
| BiasAlert | Real-time bias detection framework for judges | S3 |
| pass@k / pass^k | Probability of success across k attempts | S4 |
| Additive Scale Prompting | Award points for each atomic criterion met | S15 |
| Continual In-Context Learning | Dynamically select relevant examples per evaluation item | S11 |
| Instance-Specific Rubrics | Generate custom rubrics per input rather than using fixed ones | S17, S19 |

---

## Open Questions for the AI Egg Index

Based on this survey, key evaluation decisions we still need to make:

1. **Binary vs. scale scoring?** — Binary (pass/fail) is more reliable across sources (Hamel, Anthropic, HuggingFace all recommend it), but we lose granularity. Could use binary per-criterion with additive total.

2. **How to validate our LLM judge?** — Every source says: label 30-100 examples by hand, measure agreement (Pearson or Cohen's Kappa), iterate on rubric until correlation is acceptable.

3. **Position bias in our benchmarks?** — Not an issue for our single-output scoring, but relevant if we ever do pairwise comparisons.

4. **Statistical rigor** — We should report confidence intervals and run multiple evaluation passes (Anthropic's paired-differences approach).

5. **Contamination** — Our custom benchmarks (practical knowledge, creative+technical) are less vulnerable than standard benchmarks, but we should track knowledge cutoff dates and refresh questions.

6. **Are we measuring the right things?** — The gap between "benchmark performance" and "real-world usefulness" is a known unsolved problem (BetterBench, Stanford). Our practical-task focus is the right instinct, but we need to validate that our questions actually represent what people ask.
