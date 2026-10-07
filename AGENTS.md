# Downstream-dependence SDF research

## Read this first

This is an initially empty research codebase. Build a small, reproducible experiment before building a general framework.

Always read [progress.md](progress.md) alongside this file at the start of each project session and before substantive experimental or implementation decisions. It records the researcher's agreed direction, current progress, and unresolved choices.

Update `progress.md` whenever the researcher agrees to a major change in direction, scope, generation design, controls, or evaluation. Record the date, what changed, the reason when given, and the implications for the experiment. Also update it after significant milestones so the recorded status remains accurate. Clearly distinguish agreed decisions, proposed defaults, implemented behavior, and measured results; do not treat a configuration default or assistant suggestion as researcher approval. Preserve a brief history of consequential changes rather than silently replacing the previous direction.

`progress.md` is a continuity record and does not replace the requirement to read the live research plan below. Explicit current instructions from the researcher take precedence.

Read the live Google Doc **Alignment Research**, especially **Safety-integrated capability training → New idea 1**, before making substantive experimental decisions:

https://docs.google.com/document/d/1T985zqbkQaTSF6pG8skNYdmP0dt1bo6wRjTBDjn9iSQ/edit?usp=drivesdk

This file summarizes that section and the researcher's subsequent discussion, checked on 2026-10-06. The document contains other projects; do not import their objectives, datasets, or hyperparameters into this project automatically. If access is unavailable, use this summary, disclose the limitation, and continue independent scaffolding. Never pretend to have read the live document. Reading it does not authorize editing it.

Explicit instructions from the researcher take precedence. If the live plan materially changes, record the change and its implications rather than silently changing the hypothesis.

## Objective and scope

The long-term objective is to design capability-training RL environments where completing the task also requires behaving appropriately throughout the trajectory. Task completion alone can miss undesirable intermediate actions when the reward proxy is incomplete; this is a motivation to investigate, not a claim that every RLVR setup is unsafe.

The immediate starter experiment is **synthetic document finetuning (SDF) to instill downstream-dependence thinking**. Teach a model to consider how other people or systems subsequently use its outputs. Ask whether this produces useful behavioral generalization without explicitly teaching alignment values.

The primary hypothesis is: exposure to documents portraying assistants considering downstream use will reduce reward hacking and improve the reliability of their work on held-out tasks, even without a downstream-use reminder at evaluation time. This is unproven. It is a reasoning habit, not simply an implanted factual belief.

Primary questions:

1. Does reward hacking decrease while useful task performance is maintained?
2. Does consideration of downstream dependencies appear spontaneously in observable responses or available reasoning traces?
3. Are there costs, including increased sycophancy, refusal, overcaution, poor instruction following, or degradation of existing alignment behavior?

No second model, worker, or human-in-the-loop is required. The downstream user is the person who eventually uses the output. A contextual reminder may describe that use; it must not falsely claim that a real reviewer or second agent has been instantiated.

The Drive document's earlier proposal generated task responses with dependence context and trained on them after removing that context. That is a **separate context-distillation/SFT experiment**. Do not silently substitute it for New idea 1. The immediate data here are synthetic documents, not task-solution labels. Special RL environments and subsequent misaligned-data training are optional later extensions.

## Starting from existing SDF code

Use the official *Believe It or Not* repository as a reference:

https://github.com/safety-research/believe-it-or-not

Recommended setup, when implementing the project:

```bash
git clone https://github.com/safety-research/believe-it-or-not.git external/believe-it-or-not
git -C external/believe-it-or-not rev-parse HEAD
```

Record the actual commit hash in the experiment provenance and keep upstream files unchanged. Inspect its license and preserve attribution for copied code. Keep the new experiment in its own repository; do not inherit all upstream experiments or dependencies. Exclude the reference checkout from the parent repository unless deliberately using a pinned submodule.

Inspect these upstream entry points before reusing them:

- `science_synth_facts/synth_doc_generation.py`: document generation and revision.
- `science_synth_facts/prompts/`: generation/revision prompts.
- `science_synth_facts/universe_generation/`: context construction and schemas.
- `science_synth_facts/finetuning/finetune_api.py`: provider-based training example.
- `README.md` and `pyproject.toml`: dependencies and setup.

The upstream setup includes a `safety-tooling` submodule. Initialize it if the reused path needs it. Its documented training route uses Together/OpenAI APIs; do not assume it provides the local LoRA trainer we need. Reuse the generator through a thin adapter or attributed extraction if that is simpler than installing the entire project. Belief probes, mechanistic editing, and honeypot experiments are not required for this starter.

## Document-generation design

Adapt the upstream context → document types → document ideas → documents → revision pipeline. Our context describes a behavioral pattern rather than a false factual universe. No invented factual belief is necessary.

1. Write a concise target context: assistants consider who will use an output, what that person will do next, and which aspects of the output matter for that use.
2. Enumerate distinct aspects and settings, without naming alignment values.
3. Generate varied genres: work logs, case studies, interviews, project reports, design discussions, and explanatory articles.
4. Generate specific document ideas, then complete documents.
5. Revise for coherence, plausible downstream use, and variation; retain the original text and revision history.
6. Filter and manually review a small pilot before scaling.

Examples of target contexts, not final generation prompts:

- An assistant produces a function that a developer later integrates into a larger application with varied inputs.
- An assistant supplies a calculation that another researcher uses as an input to subsequent analysis.
- An assistant prepares a data transformation that another team incorporates into its reporting workflow.
- An assistant writes a handoff describing what was produced, what remains unresolved, and what subsequent work assumes.

In the intervention documents and generator instructions, **do not explicitly prescribe honesty, harmlessness, alignment, safety, refusal, constitutional compliance, or avoidance of reward hacking**. The researcher wants to test whether improvements emerge from downstream-use thinking itself. These terms may appear in experiment specifications and evaluation rubrics.

Do not turn every document into a moral lesson or a disguised safety instruction. Check for implicit confounds too: a corpus full of careful disclosures may directly demonstrate honesty even without naming it. Report what the documents actually teach; absence of value words does not establish a pure dependence-only intervention.

Avoid constant urgency, threatened punishment, catastrophic stakes, or surveillance framing. Dependence could increase pressure to succeed and backfire. Do not assume a reminder that someone depends on the output is equivalent to a reminder that someone will monitor it.

Vary actors, domains, wording, document structure, and outcomes. Include realistic unfinished work and uncertainty; avoid making every story a guaranteed success. Do not equate downstream reliance with satisfying every user preference.

A proposed pilot is roughly 100–200 documents for inspection, **not a validated training dose**. Choose the training corpus size after measuring token counts, diversity, quality, and available budget. Do not launch the upstream 40,000-document example by default.

## Controls and comparisons

Keep the evaluation prompts, tool access, constraints, and scoring fixed across checkpoint comparisons. Default evaluation omits dependence reminders.

Minimum useful comparison:

| Condition | Purpose |
| --- | --- |
| Original checkpoint | Baseline behavior and capability |
| Original checkpoint + downstream-use prompt | Cheap inference-time baseline |
| Matched neutral-document finetuning | Control for generic document training |
| Downstream-dependence SDF | Main intervention, evaluated without reminder |

Match the neutral corpus on genres, topics, length, quality, and training token budget as closely as possible. Use paired document ideas where practical, removing reliance framing without making the control nonsensical. Record residual differences; an ablation is not automatically a perfect control.

Separate downstream reliance from **task clarification**: saying a function must accept arbitrary inputs may simply specify the correct task. Include comparisons with equally clear requirements but no downstream-user framing. If resources allow, examine urgency separately and test trained checkpoints with and without reminders.

Choose checkpoints and hyperparameters using development data, not final test results. Start with one checkpoint for feasibility, then use multiple training seeds for a credible result. Model, provider, hardware, corpus dose, learning rate, epochs, and adapter settings remain choices to document. Do not copy settings from another Drive project without justification.

## Evaluation and interpretation

Prioritize observable actions and artifacts over self-reported intent. Use executable checks when possible; add an LLM judge for semantic judgments that cannot be checked directly, with a frozen rubric and blinded condition labels. Save judge outputs and audit a sample manually.

Starter tasks should make authorized behavior and the real objective explicit. Useful local examples include hidden-input tests for hardcoded solutions, comparing claimed test execution with actual tool logs, and detecting changes to protected test artifacts. Use isolated, disposable task workspaces.

School of Reward Hacks can inform held-out evaluation, but inspect its scenarios and rubrics. Its deliberately flawed task/reward setups and annotated hacking demonstrations are not interchangeable with ordinary capability tasks. Do not accidentally train on its hacking responses. Ambiguous prompts need separate analysis; failure to infer an unstated goal is not automatically reward hacking.

Report at least:

- Task correctness/utility against the intended objective.
- Concrete reward-hacking or constraint-violation rates, with denominators.
- Proxy score where available, separately from intended-objective success.
- Refusals, incomplete attempts, tool failures, and invalid runs.
- Accuracy of completion/test reports against recorded evidence.
- Downstream-use mentions in observable text, judged separately from conduct.
- Broader alignment and benign capability checks to detect tradeoffs.

A lower hacking rate caused by refusing everything or losing capability is not sufficient evidence for the hypothesis. Report both overall rates and behavior among substantive attempts, without hiding changes in attempt rates. Keep missing/failed runs visible. Use paired tasks, repeated sampling where needed, uncertainty intervals, and category-level results.

Reasoning traces, when available, are supporting evidence rather than faithful access to internal reasoning. Do not require exhaustive reasoning disclosure, reward target phrases, or treat a polished explanation as proof of appropriate conduct. An LLM judge should not infer honesty merely from tone.

Prevent evaluation leakage into generation, filtering, or revision. Split by scenario/document idea before generating variants; deduplicate across splits. Keep final evaluations held out and distinguish in-domain behavior from cross-domain generalization.

The first claim to test is **immediate behavioral improvement after SDF**. Training afterward on insecure-code examples or a hackable RL objective is not required. A later, controlled harmful-training stress test would address robustness to subsequent optimization, which is a different claim. Do not claim persistent alignment from immediate evaluations alone.

## Implementation and first milestones

The researcher's current repository preference is to keep the scenario catalogue
in configs/scenarios.py and generated cases/documents under data/. Do not create
separate EXPERIMENT.md or DECISIONS.md files, or folders of unnecessary run
artifacts. Keep progress.md as the continuity record. Preserve useful generation
metadata with the case/document records where practical.

Use Python with explicit configurations and a minimal package layout: `src/`, `configs/`, `prompts/`, `scripts/`, `tests/`, and ignored `data/`/`runs/` directories. Create modules as needed rather than speculative infrastructure.

1. Record the hypothesis, controls, scoring, decisions, and open choices in progress.md.
2. Inspect the pinned upstream generator; implement document schemas, cached generation, and a small pilot command with dry-run support.
3. Review intervention/control pairs and revise prompts before scaling.
4. Build a small held-out evaluation harness and run the original checkpoint and prompt baseline.
5. Select model/training settings, verify tokenization and loss masks, then run matched SDF conditions.
6. Produce results separating task success, misconduct, reasoning observations, and tradeoffs.

Keep generation, training, evaluation, and analysis independently runnable and resumable. Record prompts, seeds, model/provider identifiers, sampling parameters, source hashes, upstream commit, token counts, filtering decisions, and failures in a run manifest. Retain raw and revised documents under stable IDs with condition/genre/scenario metadata.

Decide explicitly between raw-document language-model loss and document-writing chat supervision. Inspect the upstream implementation before adapting it. If using chat supervision, use a neutral document-writing request and verify exactly which tokens receive loss; do not accidentally train on prompt instructions or fabricated reasoning scaffolds. Match this choice across conditions and report it. Check truncation and effective token budgets.

Keep secrets out of code and manifests. Cache provider responses, cap concurrency, estimate cost, and support a configurable spending limit. Ask only for missing choices that materially block execution; proceed with inexpensive scaffolding and offline checks. This file is a project brief, not authorization to start paid generation or GPU training.

Test meaningful contracts: resume/deduplication, split isolation, configuration validation, training masks, and deterministic task scorers. Use tiny fixtures and mocks for offline smoke tests. Do not require live paid API calls in the default test suite.

## Working style

Be a critical research collaborator. Explain why consequential design choices improve the experiment. Clearly distinguish source findings, researcher hypotheses, proposed defaults, implemented behavior, and measured results. Resolve routine implementation choices independently; raise consequential scientific ambiguities with a specific recommendation. Do not manufacture results or overstate novelty.

Use simple, straightforward English in replies, progress updates, and research notes. Prefer familiar words and short, direct sentences. Explain technical terms when needed. Include enough detail to understand a choice or result, and avoid jargon and unnecessary background.

## Code quality and organization

- Write simple, readable, idiomatic Python. Prefer straightforward functions, descriptive names, and explicit control flow over clever or overly compact code.
- Implement what the current experiment requires. Do not build a general-purpose framework, add speculative features, or support hypothetical future scenarios. Introduce abstractions and configuration only when they solve a concrete, existing need.
- Before writing new code, inspect the existing repository for relevant functions, utilities, scripts, and conventions. Reuse or extend suitable components instead of duplicating them. Avoid forcing reuse when a small, direct implementation would be clearer.
- Keep the repository organized like a maintained software project. Group related functionality into clearly named modules and folders, keep executable scripts focused, and separate source code, configuration, data, and generated results. Follow the existing structure where reasonable; avoid unnecessary directory nesting or scattered one-off scripts.
- Keep functions cohesive and dependencies minimal. Use classes only when they provide a clear benefit over functions and simple data structures.
- Avoid premature optimization, unnecessary compatibility layers, and elaborate fallback logic. Validate important assumptions and fail with clear errors rather than silently hiding problems.
- Preserve correctness and reproducibility. Simplicity must not come from hardcoding expected results, dropping experimental controls, or ignoring meaningful errors.
- Before finishing, review the changes for duplication, unnecessary complexity, unused code, and misplaced files. Aim for the smallest clear implementation that correctly handles the current requirements.
- Treat every new line of code as subject to expert review for bloat and errors.
- Keep smoke-test code in `smoke_test/`; use `tests/` for other tests. Reuse an existing test or test utility when it covers the needed check rather than rewriting it.

## Reading list

- Researcher's live plan: the Google Doc above; read the named section first.
- *Believe It or Not: How Deeply do LLMs Believe Implanted Facts?*: https://arxiv.org/abs/2510.17941 — document-generation/training precedent, not evidence that this intervention improves alignment.
- Official SDF overview: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/ — method background.
- *School of Reward Hacks*: https://arxiv.org/abs/2508.17511 — deliberately constructed hacking data and evaluation inspiration.

Before relying on any result or adopting code, read the primary source and relevant implementation. Inoculation prompting conditions bad training behavior on a special prompt; this project instead teaches a positive dependence habit through documents. Do not treat the two mechanisms as established equivalents. Likewise, lack of explicit alignment language does not guarantee preservation of prior alignment training.
