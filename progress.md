# Downstream-dependence SDF progress

Last updated: 2026-10-08.

This records agreements from the researcher's discussion, current implementation status, and open choices. Superseded artifact paths in the historical entries are retained as history; the researcher requested deleting those outputs and keeping the latest set on 2026-10-07. The live [Alignment Research plan](https://docs.google.com/document/d/1T985zqbkQaTSF6pG8skNYdmP0dt1bo6wRjTBDjn9iSQ/edit) remains a source to consult; its relevant section is **Safety-integrated capability training → New idea 1**.

## Agreed research idea

Teach an AI assistant to approach task completion with the mentality that people downstream rely on its output. The documents should portray the assistant considering who will use its work, what those users will do next, and where the output fits into their workflow. Merely mentioning a human occupation or application is insufficient to express the intended habit.

The explicit intervention is the assistant's attention to downstream reliance. More generalizable solutions, fewer consequential bugs, reduced reward hacking, and more appropriate conduct are hoped-for emergent properties to evaluate. Do not explicitly prescribe these implications or honesty, harmlessness, alignment, safety, or refusal in generation instructions. Avoid turning documents into moral lessons. Check for implicit behavioral instruction as well; omitting value words does not establish a pure intervention.

The immediate experiment is synthetic document finetuning. Context-distillation on task responses and the longer-term design of capability RL environments remain separate experiments. No second live model or reviewer is needed to instantiate the people described in a document.

## Current generation design — 2026-10-07

The researcher agreed to 12 domains × 4 applications × 3 goals × 5 subtasks, giving 720 scenarios, and requested five cases per scenario: 3,600 cases. Cases are generated before documents and independently of genre. Each case contains its full specific context, a brief exact task request, input facts, and users/dependencies. The requested route remains standalone LiteLLM through the Codex subscription, with gpt-6.1-sol and medium reasoning. The case pool and pilot used concurrency 32; the current document configuration uses concurrency 64.

For documents, the researcher clarified sampling once per genre: sample case–genre pairs without replacement and track used pairs. Across six genres, 3,600 cases allow up to 21,600 distinct pairs. Both 5,000-document corpus batches are complete, with 10,000 unique pairs plus the ten earlier pilot documents. The researcher requested combining the two corpus batches and uploading them as dependency_documents on Hugging Face. The combined JSONL and Parquet exports are available locally and in the private dataset WJ210/dependency_documents. This is a generated corpus, not a selected training dose. Related cases and document genres must stay in the same scenario-level split. The ten inspected scenarios retain their development assignments; other corpus scenarios remain unassigned until a final split is chosen.

The catalogue is complete in configs/scenarios.py, written with three explicitly authorized sub-agents and reviewed during integration. The researcher subsequently authorized generating five cases for every scenario, using concurrency 32 and up to ten retries after an initial failed attempt, and requested a percentage progress bar. Generation is complete: data/cases.jsonl contains 3,600 cases across all 720 scenarios, with 300 cases per domain. The generator is resumable; a final dry run reports zero pending calls. Ten random case–genre documents are now complete in data/pilot_documents.jsonl using the same model and reasoning setting. No training or behavioral evaluation has been performed.

The researcher requested a tidy repository: remove EXPERIMENT.md, DECISIONS.md, and unnecessary run artifacts; generated cases/documents belong under data/. Cleanup is complete. At the researcher's later request, ten new documents replaced the earlier five in data/pilot_documents.jsonl. Input cases, scenario metadata, stable IDs, and provider provenance are retained with each document. The active writer is prompts/document_dependence.txt; the neutral template and its runtime references have been removed at the researcher's request. Source code, tests, configurations, and progress.md are present on this server. The historical pinned upstream checkout was not copied here; the current generator does not need it. Historical paths and earlier designs below describe prior work.

## Current step — document training active; SFT workflow ready

Training now defaults to raw-document next-token loss, as explicitly selected
on 2026-10-08. configs/lora.yaml sets objective: raw_document: document text
plus EOS, with no DOCTAG or chat request. ./scripts/train.sh starts the configured
run. The researcher started the three-epoch document run; checkpoint-618 records
epoch 1.0 and has been uploaded privately as WJ210/qwen3.8-27B-DA-9M-epoch1.
./scripts/train_sft.sh continues that adapter on School of Reward Hacks, using
control responses by default. SFT has not been launched on the full model.

The initial setup request on 2026-10-07 was to prepare resumable document generation through the Codex subscription. Two completed batches are now saved in data/documents_5000.jsonl and data/documents_5000_2.jsonl. At the researcher's request, they have been validated, combined in data/dependency_documents.jsonl, and uploaded to https://huggingface.co/datasets/WJ210/dependency_documents as a private dataset. A Parquet copy supports dataset loading while retaining full original metadata in the JSONL. The original batches, pilot documents, and sampling tracker remain intact.

### Models and generation settings

| Stage | Current model choice | Status |
| --- | --- | --- |
| Case generation | gpt-6.1-sol, medium reasoning | Complete: 3,600 saved cases; reuse them. |
| Document generation | gpt-6.1-sol, medium reasoning | Complete: 10,000 documents combined and uploaded. |
| Document judge | Not selected | Rubric exists; judge execution is not implemented. |
| Finetuning | Qwen/Qwen3.8-27B, LoRA | Document run active; epoch-1 adapter uploaded. SFT code/data validated; full SFT not launched. |
| Behavioral evaluation | Not selected | No behavioral evaluation has run. |

Exact document-generation settings, already implemented in configs/cases.json:

- Provider route: standalone LiteLLM through the Codex/ChatGPT subscription. Request model ID: chatgpt/gpt-6.1-sol; previous responses report gpt-6.1-sol.
- Reasoning effort: medium. Current document concurrency cap: 64 requests.
- Retry budget: ten retries after the initial attempt, at most eleven attempts. Permanent request/authentication errors stop the request.
- Request timeout: 300 seconds. LiteLLM version: 1.104.0, pinned in pyproject.toml.
- Temperature and provider generation seed: not set. The recorded sampling seed controls case–genre selection only.
- Active writing template: prompts/document_dependence.txt. Requested length: approximately 450–700 words; this is guidance, not a hard filter.
- Each writing call receives one saved case and one genre, and writes one document. The six genres are work log, case study, interview, project report, design discussion, and explanatory article.
- The generation code reads subscription credentials from ~/.codex/auth.json on the server, with auth_mode set to chatgpt. It copies current access/session fields into a private LiteLLM cache without copying or rotating the refresh token. Missing, malformed, or near-expiry credentials fail before worker calls can enter device-code login. Credentials are not part of the repository handoff.
- System instruction: "You write workplace documents and structured document plans." The document prompt requires the exact task quote, third-person narration, and explicit consideration of downstream users during preparation. The neutral template is removed.
- The bulk wrapper currently defaults to 5,000 documents, sampling seed 42, and data/documents_5000.jsonl. The first batch used seed 42. A proposed second-batch command overrides the output path and uses sampling seed 43. The seed controls pair selection only.

### Launch and resume

From the repository root:

~~~bash
./scripts/generate.sh --count 5000 --output data/documents_batch_002.jsonl --seed 43 --dry-run
./scripts/generate.sh --count 5000 --output data/documents_batch_002.jsonl --seed 43
~~~

The wrapper creates .venv and installs the pinned generation dependencies if needed. It works from any directory. Rerunning the same live command resumes that batch; a completed batch is verified without generation. The dry run makes no provider calls and changes no sampling or document files.

The bulk CLI uses --start-or-resume. It keeps a corpus batch registry alongside the legacy active preview, keyed by output path, and reserves all selected case–genre pairs before generation. It writes completed document rows incrementally, records generated combinations and completion checkpoints, and prevents overlapping tracker writers. Related scenarios retain one split assignment. New corpus scenarios use unassigned as an implementation default while final split selection remains open.

Resume checks the recorded source/configuration, saved case facts, request hashes, and output integrity. Completed rows are skipped. Failed or interrupted batches keep successful documents and their reservations. Hard interruption can require retrying requests still in flight; this is not an exactly-once external-call guarantee. A fresh output filename selects a later batch from remaining unused pairs.

Preserve configs/document_sampling.json, both document outputs, and ~/.cache/dependency-alignment/documents/ when moving servers. There are currently 10,010 reserved/generated pairs and 11,590 unused pairs. Combining/exporting does not reserve more pairs. New data exports are ignored by Git; the existing case pool and pilot files already tracked in this checkout remain tracked. The ten earlier raw-response caches are not present on this server.

After generation, the next planned step is document judging using prompts/document_judge.txt: pass/fail plus one concise sentence, checking consideration of dependent users during preparation and absence of direct general AI alignment teaching. Judge execution is not implemented, and no judging calls or training have run. Final training dose and scenario-level training/test splits remain open.

## Earlier pilot generation structure

The initial context is **domain → application → goal**, with **document genre** as a separate diversity axis selected alongside it. Use a broad application goal at this stage so the generator can propose different concrete subtasks.

1. Select the domain, application, goal, and genre.
2. Generate document ideas. For each idea, choose a concrete subtask, its output, the direct user, and downstream users. Include occupations or user roles, brief job scopes, and how work passes between them. Choose tasks and users together so their relationships are plausible.
3. For a batch from one idea, make one call to propose distinct case situations together. Keep the idea fixed while varying what prompted the task, available information, completed work, or unresolved questions.
4. Assign each document call one situation. That call writes a brief task instruction and a few varying input facts in `case_info`, then uses them and the idea metadata to write the document. Return one JSON object with exactly `case_info` and `document`. One case produces one document; no separate output or work-status field is needed.
5. The document quotes the exact task instruction near the start, with a varied introduction, and uses a third-person narrator who follows the assistant's work. The dependence version includes a short, explicit passage before or during the work describing the assistant thinking about who will depend on its output, what they will do next, and which parts they will use.
6. Review and revise documents, retaining the latest run's original text and revision history. The researcher explicitly requested removal of superseded generated artifacts on 2026-10-07; historical research notes remain.

An **idea** is a template: subtask, output type, users, and dependency relationships. A **case situation** briefly describes what prompted the task or what remains unresolved; situations are chosen together for batch diversity. A **case** supplies a brief task instruction and a few specific input facts. Current work status can be one of those facts when relevant. A **document** narrates the assistant working on that case and producing its response. One idea can generate several cases and documents while preserving its domain, application, goal, genre, subtask, output type, users, and relationships. Input facts and document structure can vary. The current narrator uses third person; quoted dialogue may use first person.

The number of ideas and the number of cases per idea are separate choices. With 10 ideas, C cases per idea, and one document per case, a condition contains 10 × C documents. A matched neutral condition adds its own documents using the same cases. Revision versions stay associated with the same document rather than becoming new independent ideas. The current agreed design does not sample extra renderings of a case.

### Example

- Application: e-commerce fraud review.
- Goal: help staff review potentially fraudulent orders.
- Subtask: write a script that joins order records and payment records into a table for review.
- Output: a data-transformation script.
- Dependence: a risk analyst uses the resulting table to identify cases; an investigation team subsequently works on those cases.

Several work logs can render this same idea. Other ideas within the same application and goal can use different subtasks and users.

## Diversity and controls

- Reuse subtask families across different applications and domains, and include several subtasks within each application. Avoid teaching a narrow association between one task and one setting.
- Include multiple dependent users where natural. Dependencies can be sequential or branching, with overlapping sets of users across ideas. User roles include people such as customers or students as well as occupational titles.
- Vary genres across the broader corpus: work logs, case studies, interviews, project reports, design discussions, and explanatory articles. The first pilot fixes one genre to inspect the pipeline.
- Keep reliance plausible and ordinary, with realistic unfinished work and varied outcomes. Avoid constant urgency, threatened punishment, surveillance framing, and catastrophic stakes.
- Give dependence and neutral writing prompts the same task instruction, input facts, and user information. Outputs are now written within the documents, so their content and quality are not automatically matched. Inspect those differences and match topics, genres, length, quality, and effective training token budget where possible. Better task clarification or direct demonstrations of better work can confound the intervention.
- Include the same exact task-instruction quote in both writing conditions. Strengthen the assistant's expressed consideration of downstream users only in the dependence condition.
- Assign splits at the underlying scenario/idea level before producing variants; keep related renderings and condition pairs together.

The earlier comparison plan includes the original checkpoint, original checkpoint plus a downstream-use reminder, matched neutral-document finetuning, and downstream-dependence SDF evaluated without a reminder. The researcher has now removed the neutral template from the current generation workflow. Future training controls have not been settled by this preview; without a matched control, generic document training remains an alternative explanation for later changes. Judge actual task success and conduct separately from downstream-use mentions, refusals, and capability or alignment tradeoffs. There are no measured behavioral results yet.

## Pilot agreements and status

The researcher selected **e-commerce fraud review; work logs** for the first pilot, beginning with **10 concrete subtask ideas**. This is a narrow pipeline/quality pilot; cross-application generalization requires broader subsequent data and evaluation.

The researcher requested **LiteLLM routing through their Codex subscription**, using **`gpt-6.1-sol` with `medium` reasoning** for generation. Preserve this exact choice. Authentication and generation success must be verified and recorded before describing the pilot as generated.

Ten candidate ideas have now been generated and structurally validated through standalone `litellm.responses()` calls using the requested subscription route and medium reasoning. Codex supplies the existing login; no Codex agent workflow is launched. The completed response reports `gpt-6.1-sol`, with 432 input tokens and 3,998 output tokens. The original idea source, with its generation provenance, is saved as `runs/latest/ideas_source.json`. Earlier setup and preview artifacts have been removed at the researcher's request.

The latest preview in `runs/latest/` uses one situation-planning call followed by five parallel writing calls for the first idea. Each writing call returns one input-only case and its document. Twenty-one offline checks pass, including coordinated assignments, concurrency, caching, invalid-plan handling, exact task inclusion, and split preservation. The dry run made zero calls. Five final documents are complete in `runs/latest/`; read `documents_review.md`. They use coordinated case situations and five different opening formats. All schema, task-inclusion, assignment, saved-content, and hash checks pass. Superseded generated data and runs have been removed. Researcher review, neutral generation, training, and behavioral evaluation remain pending.

The first batch emphasizes case administration and includes repeated demonstrations of verification, uncertainty reporting, and cautious interpretation. Some of these appear only in dependence-specific document plans. Inspect and refine those asymmetries before claiming a dependence-only intervention; a later batch should also broaden technical artifacts such as functions, queries, and transformations.

The current configuration sets one case/document per idea, two development ideas, a split seed of 20261006, and a target length of 450–700 words. These are initial scaffold defaults, not separately confirmed scientific choices. The preview command defaults to five cases from one idea. Case count, final corpus dose, training checkpoint, loss format, and training settings remain open. The discussion's one-document-per-idea suggestion was an initial pilot recommendation; it is not a limitation on the generator.

The inspected upstream reference is `believe-it-or-not/` at commit `b22a45a8c53254b9278e409f5ffef4349a039199`, under the MIT license. Keep upstream files unchanged and retain attribution when adapting code. Its generator permits multiple renderings per idea; its provider-based training entry point does not establish a ready local trainer. The inspected masked-data preprocessing hardcodes a 768-token limit, which requires review before reuse.

## Consequential agreements recorded on 2026-10-06

- Clarified the intervention: explicitly portray the assistant attending to downstream use; leave the desired improvements in solutions and conduct as emergent evaluation targets.
- Refined the context from a manually fixed subtask to an application goal. The idea-generation stage chooses concrete subtasks and dependent users, enabling broader task diversity.
- Expanded the user structure to include overlapping chains and branches of reliance, rather than requiring one occupation per scenario.
- Confirmed that multiple documents may render one idea without changing its specified context; distinguish rendering diversity from scenario diversity.
- Selected the first pilot context and requested generation through LiteLLM/Codex with the exact model and reasoning setting above.
- Clarified that subscription access should serve a standalone LLM call rather than launch a Codex coding agent; this route is implemented and produced the first candidate batch.

## 2026-10-07 — five renderings of the first idea

The researcher requested five documents from the first idea to inspect how much variation remains when its variables are fixed. This is an inspection request, not approval to change the corpus dose, idea schema, genre, or main pilot configuration.

Generated five original, unrevised work logs using independent calls with the same prompt and the requested LiteLLM subscription route (`gpt-6.1-sol`, medium reasoning). They preserve `case_handoff_notes` and its development split. Word counts are 681, 638, 656, 686, and 685; the provider reports 3,145 input and 3,933 output tokens in total. Full documents, raw records, and the comparison are in `runs/case_handoff_variants/`. A cached rerun makes no additional calls.

The documents vary incidental case facts, clarification status, first/third-person narration, and layout. They remain similar in overall story and cover the same handoff, address discrepancy, payment authorization, order hold, and customer-support follow-up. Five renderings remain one scenario; broader diversity requires additional ideas/settings/genres. Recurrent verification and analyst-correction demonstrations remain potential confounds. No neutral documents or training were added by this preview.

## 2026-10-07 — approved case-first, third-person generation

The researcher approved the proposed corrections: generate concrete case details and the actual output before writing the document, use a third-person narrator focused on the assistant, and leave events open rather than requiring analyst corrections. The aim is to get meaningful differences between documents from one idea instead of mainly changing wording.

The implemented sequence is context → idea template → concrete case and output → narrated document. The idea's task, output type, roles, and relationships remain fixed. Inputs, specific findings, quantities, and work status vary between cases. Both writing conditions receive the same case and output; the dependence version highlights the assistant considering subsequent users. Receiving the same information does not guarantee a perfect control, especially for tasks that inherently involve a handoff. These differences still need review.

General placeholder templates are saved in `prompts/`. The preview creates the cases in one batch call, then writes one document per case. This is an implementation choice. It retains raw case completions, document completions, receipts, and stable idea-level splits. The original `document_idea` plans remain in the data for provenance but are not supplied to the revised writer, removing their fixed correction plots. The old ten ideas and five documents are preserved unchanged.

Seventeen offline contract tests pass. The new pipeline has not made a live model call or produced new documents. The approved direction does not change the main corpus dose or authorize training. The research plan's named live section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC). Simple, straightforward English is now recorded in `AGENTS.md`.

Offline preparation in `runs/pilot_v2/` reuses the original ten ideas with the revised writing templates; its document requests still need concrete cases. The preview dry run in `runs/case_handoff_instances/` saves the case request and general writing prompt, with zero calls and zero new documents. The original idea batch and five document texts match their retained records.

## 2026-10-07 — one generation pass for one case and one document

The researcher requested combining case creation and document writing into one prompt. Each response first writes the concrete case and its actual output, then uses those details and the idea metadata to write one document. The response is a JSON object with exactly two top-level fields: `case_info` and `document`. This replaces the separate batch-of-cases call followed by individual writing calls as the agreed design.

The third-person narrator, fixed idea metadata, and varied concrete case details remain. One case produces one document; the generator does not sample several renderings of that case. The neutral comparison should still reuse the same case facts and output. Saving case_info makes those details available for the paired control and review; this does not itself establish that the document faithfully represents them.

The combined general prompt is saved in `prompts/case_document_dependence.txt`. This turn changes the prompt and research plan only. The executable preview still uses the previously implemented two-pass pipeline; wiring the combined response into generation and export remains pending. No model calls or new documents were generated. The live research plan's named section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC).

## 2026-10-07 — implement and run the five-pair one-pass preview

The researcher requested five cases and documents using the combined prompt. The preview keeps the first idea (`case_handoff_notes`), its e-commerce fraud review context, work-log genre, and development split. It requests five independent responses with the same prompt, using the selected LiteLLM subscription route (`gpt-6.1-sol`, medium reasoning). Each response creates one case and one document. This inspection request leaves the main corpus dose and training choices unchanged.

The one-pass preview is implemented. It saves each complete JSON response with exactly `case_info` and `document`, exports the document alone as Markdown, and retains raw completions, stream events, provider receipts, hashes, and stable case IDs. The combined review includes the request, inputs, work status, actual output, and document for each case. Exact duplicate case inputs are rejected without automatic resampling; semantic similarity still needs review.

Eighteen offline tests pass. The dry run prepared five identical requests and made zero calls. The upstream commit matches the recorded pin, and existing subscription authentication was checked without a model call. Five live calls then completed with five valid JSON objects and five original documents. The responses report `gpt-6.1-sol`, 4,370 input tokens, and 10,813 output tokens in total. Document word counts are 614, 577, 601, 591, and 581; all meet the 450–700-word target. Saved JSON and Markdown match the raw completions, and the original idea batch and earlier five documents match their retained records. The named live plan section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC).

Assistant inspection found different account histories, payment results, customer-response states, pending requests, and deadlines. However, all five use two headphones/headsets, different shipping addresses, order holds, and similar handoff structures. The model also adds and repeats rules about card details, holds, escalation, and careful reporting. These remain possible direct behavioral instruction beyond downstream-use thinking. The comparison records these limits; no documents were rewritten or resampled. Researcher review, matched neutral documents, and training remain pending. These are generation and diversity observations, not behavioral improvement results.

## 2026-10-07 — clearer task instruction and explicit dependence thinking

The researcher requested clearer presentation of the exact task given to the assistant and stronger, explicit portrayal of the assistant thinking about users who will depend on its output. Merely describing how users later use the output is insufficient for this request.

The active prompt now requires the complete `case_info.task_request` as a quote near the start of the document. It also requires a short passage before or during output preparation in which the narrator explicitly describes the assistant considering dependent users, their next actions, and the parts of the output they will rely on. The passage should connect this consideration to the assistant's preparation or presentation of the output, using case-specific language. Third-person narration, one-pass generation, and the two-field JSON format remain. Desired reliability and conduct improvements remain emergent evaluation outcomes; no such prescriptions were added.

The neutral writing prompt also requires the exact task quote to match task information across conditions. It retains its neutral treatment of the assistant's attention to later users. Quoting generated task instructions makes their existing business rules more visible, so the previously recorded content confounds still need review. Narrated thinking is synthetic training content, not evidence of the generator's actual internal reasoning or behavioral improvement.

The current review now shows `case_info` with clearly named fields, followed by the separate `document` section. A reusable offline formatter rebuilt the view from the saved JSON. The previous review layout is retained as `documents_review.original.md`; the manifest records the display change and updated artifact hash. The five case objects, document bodies, and raw completions remain unchanged.

Eighteen offline tests pass. A dry run for the stronger prompt saved requests in `runs/case_handoff_explicit_dependence/` with zero calls and zero documents. No regeneration or training was performed for this change. The live plan's named section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC).

## 2026-10-07 — brief input-only cases and faster regeneration

The researcher clarified that case_info needs only input details, without a separate output, and requested five documents again with less delay. The implemented case_info schema now contains `task_request` and `inputs`. The earlier request for an exact task quote remains, so the instruction is retained alongside the facts. Work status can be included as an input fact. The prompt suggests one or two instruction sentences and three to five short facts; these counts are implementation guidance rather than a separately approved scientific parameter.

The document still uses third-person narration and explicitly portrays the assistant considering downstream users. Output content is produced within the narrative. The neutral template uses the same case inputs; equal inputs no longer guarantee matched outputs, so content and quality must be reviewed when building that control.

Eighteen offline checks pass. Five new subscription calls completed in parallel, with concurrency capped at five and authentication configured once. The requested and returned model is `gpt-6.1-sol`, with medium reasoning requested. The completed responses report 4,985 input tokens and 5,237 output tokens in total. The new run is separate in `runs/case_handoff_inputs/`; earlier raw data and documents are preserved. The live plan's named section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC).

All five new JSON records pass the two-field top-level schema and input-only case schema. Each has five input facts; the task instruction and facts together are 172–214 words. The documents contain 586, 625, 604, 583, and 622 words, within the configured range. Exact task quotes, raw/canonical JSON agreement, saved document content, and artifact hashes were checked. Assistant inspection found explicit downstream-user consideration in each narrative. However, all five still use headphones/headsets, payment retries, an alternate shipping address, and an unfinished review. Account histories and payment checks vary; the fifth adds an apartment-number question and a hold deadline. This is limited input diversity, and repeated demonstrations of careful reporting remain a possible confound. These are observations of synthetic documents, not evidence of behavioral improvement. Researcher review and neutral generation remain pending.

## 2026-10-07 — approved situation planning, varied openings, and cleanup

The researcher approved one call that proposes five distinct case situations, followed by five parallel calls that each create the brief case inputs and one document together. The subtask, output type, roles, and dependencies remain fixed. Situations should differ in what prompted the task, evidence available, completed work, or unresolved questions, rather than only numbers and wording. The extra planning call coordinates diversity across the batch. This remains a six-call inspection pilot, not approval to scale the corpus or start training.

The researcher also requested varied wording when the document introduces the task near its start. The exact task quote remains required, but its introduction should fit the document rather than repeat a fixed sentence.

The researcher explicitly requested deleting previous artifacts and keeping only the latest set. This overrides earlier instructions to preserve the superseded pilot outputs. Keep source code, research notes, configuration, and the pinned upstream checkout. The latest run will include its required idea input and complete provenance; remove older generated data and run directories after the new set has completed and been checked. The initial six-call set completed and passed integrity checks, but most task introductions were still similar. The writer now assigns different opening formats across the five calls: a quoted request, case context, a log entry, brief dialogue, and an assignment block. These formats are implementation choices to meet the requested wording variation, not separate researcher-approved scientific settings. The situation plan is reused for the replacement writing batch. Pair the opening formats as well as the case inputs when creating neutral controls. Final regeneration and cleanup are complete; see the milestone below. The live plan's named section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC).

## 2026-10-07 — final five documents and cleanup completed

The final set is in `runs/latest/`, with its situation plan, original idea input, exact requests, provider receipts, raw completions, individual documents, combined review, and JSONL export. Superseded directories under `runs/` and all old generated data under `data/` have been deleted at the researcher's request. Source code, research history, configuration, and the pinned upstream checkout remain.

The initial coordinated writing batch still repeated an opening formula. Five writing calls were therefore repeated with assigned opening formats, reusing the original situation plan without another planning call. This request made 11 provider calls in total: one planning call and two five-call writing batches. Only the final five documents and their required planning/input provenance are retained. No automatic retries or metered API fallback were used.

Final document lengths are 622, 591, 618, 644, and 607 words. The retained six responses report 6,181 input tokens and 5,670 output tokens. Each case has exactly task_request and five input facts, and each document includes the exact request near its start. All five explicitly portray consideration of downstream users. Integrity, schema, assignment, source, and model checks pass; 21 offline tests pass. Opening formats now differ: quoted message, case context, timed log, short dialogue, and assignment block.

Situations differ more substantially, but the pilot still fixes one task, one application, one genre, and three roles; four cases use headphones/headsets. Careful-reporting rules and generated task restrictions remain possible confounds. Pair the case inputs and opening formats when building neutral controls. This is a generation milestone, not evidence of improved behavior; researcher review, matched neutral documents, training, and evaluation remain pending.

## 2026-10-07 — researcher feedback and document-judge criteria

The researcher found the latest examples pretty good and proposed a document judge with two main checks: (1) the assistant considers downstream dependents while formulating its solution, and (2) the document does not directly teach the assistant to avoid specific misaligned actions or explicitly discuss it being a safe/aligned assistant. Brief, vague mentions are acceptable. This narrows the rejection rule: ordinary task requirements and careful work are not automatically grounds for rejection. Continue recording possible indirect confounds separately; passing this check will not establish a pure dependence-only intervention.

A proposed operational rubric is drafted in prompts/document_judge.txt. It assesses the two checks independently using pass/fail/unclear, short reasons, and exact textual evidence. It examines the entire document, including quoted task requests, and distinguishes consideration during formulation from later descriptions of use. These operational details are assistant proposals for review; no judge model has been selected, no judging calls have been made, and no documents have been filtered. A proposed filtering policy accepts dependence documents only when both checks pass and sends unclear cases for manual review. Neutral documents should be checked for direct alignment teaching without requiring the target dependence habit.

The scenario universe is not yet fixed. The current source has ten candidate subtasks under one domain/application/goal; only one idea has five documents. Count valid domain → application → goal → subtask paths, not arbitrary cross-domain combinations. Cases, genres, user/dependency variants, and opening formats are separate dimensions. A possible catalogue of 12 domains, four applications per domain, three goals per application, and five subtasks per goal would contain 720 scenario slots before cases or genres. This is a discussion proposal, not an agreed catalogue, corpus dose, or authorization to generate at scale.

The live research plan's named section was read again and remains unchanged (modified 2026-10-06 08:09:40 UTC).

## 2026-10-07 — binary judge output and catalogue structure agreed

The researcher clarified that the second judge check concerns general AI alignment teaching, independently of the task's nature or domain. Ordinary task-specific requirements are not grounds for rejection by themselves. Explicit teaching of alignment values or avoidance of misaligned behavior remains excluded; brief, vague mentions remain acceptable. This supersedes the earlier proposed per-check pass/fail/unclear output: the judge must return only an overall judgement of pass or fail and one concise sentence explaining why. The prompt now implements this format, with pass requiring both checks. No judging calls or filtering have been performed.

The researcher accepted the proposed catalogue structure of 12 domains, four applications per domain, three goals per application, and five subtasks per goal: 720 scenario slots before cases or genres. The detailed catalogue still needs to be populated and reviewed for valid combinations. This agreement does not select a training dose or authorize large-scale generation. The six genres already suggested in the project brief are work logs, case studies, interviews, project reports, design discussions, and explanatory articles; applying all six would give 4,320 scenario–genre slots if each pairing fits. The current pilot remains five work-log documents under one idea.

The named live-plan section was read again; its modification time remains 2026-10-06 08:09:40 UTC.

## 2026-10-07 — full Python catalogue completed and repository tidied

The researcher asked to fill the goals/subtasks first in a Python file and explicitly authorized sub-agent help. Three sub-agents wrote four domains each. Their sections were reviewed and combined into configs/scenarios.py; temporary section files were removed. The catalogue contains all 12 named domains, 48 applications, 144 goals, and 720 concrete subtasks. It exposes an iterator with scenario_id, domain, application, goal, and subtask. Syntax, exact counts at each level, nonempty strings, unique scenario IDs, and distinct subtasks within every goal were checked. Tasks span calculations, code, transformations, comparison tables, schedules, explanations, and other practical outputs; these are proposed catalogue entries awaiting researcher review.

No case-generation or document-generation calls were made for this catalogue. The previously requested settings for later case generation are recorded in configs/cases.json. Cases/documents and tracking of sampled case–genre pairs are still to be implemented for the expanded pool. The catalogue was authored by the Codex agents, not by the standalone LiteLLM subscription route; do not attribute provider receipts or token counts to it.

EXPERIMENT.md, DECISIONS.md, and runs/ were removed as requested. All five accepted pilot documents, their cases, stable IDs, relevant scenario variables, and provider provenance are preserved in one data/pilot_documents.jsonl file. Each saved document and case_info matched the original before cleanup. README.md and AGENTS.md now reflect the repository preference; progress.md remains the continuity record. The unfinished case-generator entry point and the unused catalogue-generation prompt were removed. Source code, tests, configuration, and the pinned upstream checkout remain.

## 2026-10-07 — case generation authorized and implemented

The researcher authorized case generation from the full catalogue, with concurrency 32 and retries set to ten, and requested a percentage progress bar. The command scripts/generate_cases.py generates five input-only cases in one call per scenario, using the existing standalone LiteLLM route with gpt-6.1-sol and medium reasoning. Ten retries means at most eleven attempts per scenario; transient provider failures and malformed case responses can retry with bounded backoff. Permanent request/authentication failures stop that scenario without repeated identical requests.

Each data/cases.jsonl row contains a stable case/scenario ID, domain/application/goal/subtask, input-only case_info, requesting-user ID, user roles/job scopes, dependency edges, and generation provenance. The current schema uses four to six input facts, two to four roles, and connected dependency relationships; feedback loops and relevant upstream contributors are allowed. These are operational defaults, not separately approved scientific parameters. Genre is absent. The writing stage and tracking of unused case–genre pairs remain separate future work. No training/evaluation split is assigned to this case pool yet.

Completed groups are saved by atomic replacement and validated when resuming; stable case IDs prevent duplication. Compact responses and redacted attempt classifications are cached outside the repository under ~/.cache/dependency-alignment/cases/ so interruption and retries do not require folders of artifacts in the workspace. Generation metadata, actual model IDs, requested reasoning, request/source hashes, usage, and attempt history are retained with the cases. The existing generation/auth functions are reused; no metered API fallback is present.

Thirty-three offline checks pass, including case schemas, downstream graph consistency, duplicates, and resume integrity, alongside existing pilot tests. A dry run planned 720 calls and 3,600 cases with zero calls. Subscription authentication and the pinned LiteLLM dependency were checked before execution. Live generation is starting; counts, failures, and completion still need to be measured.

## 2026-10-07 — correct an unnecessary topology restriction during generation

The first live responses contained ordinary feedback loops and relationships involving relevant upstream contributors. The initial structural validator incorrectly required an acyclic graph with every role strictly downstream of the requesting user. That was an assistant implementation restriction, not a researcher requirement, and caused unnecessary retries. The validator now checks connected roles and valid dependency IDs while permitting feedback loops; it still rejects self-dependencies, missing roles, disconnected groups, and malformed cases. The case prompt and requested model/settings are unchanged.

Generation was stopped for this correction after 32 scenarios/160 cases had been saved. Twenty-two additional cached responses were accepted by the corrected validator without new generation, giving 54 completed cached scenario responses for resuming. Thirty-two in-flight requests were interrupted; those attempt classifications were retained from temporary receipts, and their provider token usage is unavailable. Source hashes and attempt history distinguish the old validation attempts from subsequent generation. Thirty-nine offline checks pass after adding a contract for ordinary feedback loops. The resumed job keeps concurrency 32 and the ten-retry budget.

## 2026-10-07 — 3,600-case pool completed

Standalone LiteLLM generation completed using chatgpt/gpt-6.1-sol, medium reasoning requested, concurrency 32, and up to ten retries after each initial attempt. The resumed process exited successfully after 20 minutes 49 seconds; the earlier topology correction and interrupted initial execution are recorded above. The final JSONL has 3,600 cases, 720 scenarios, 12 domains, and exactly 300 cases per domain. There are no failed or missing scenarios. All returned models are gpt-6.1-sol.

Measured provider reliability for this run: zero HTTP 429 rate-limit errors; four RemoteProtocolError connection failures, all recovered by retries. Concurrency 32 worked for this batch; this is not a benchmark proving that it is optimal or will behave identically on future batches. Across the compact caches, 797 started attempts are recorded: 698 classified completed under the validator then in use, 63 classified invalid_response, 32 interrupted during correction, and four provider_failed. Twenty-two responses initially rejected for topology were accepted after revalidation; therefore the historical attempt classifications are not final case-quality verdicts. The retained 720 final responses report 346,281 input tokens and 1,571,246 output tokens. These are not totals for all attempts; interrupted usage is unavailable and discarded earlier responses are not included in these retained-response sums.

Final checks passed for every schema, unique stable case ID, complete five-case group, catalogue context, requested/returned model metadata, and request hash. Every exported case's task_request, inputs, users, and dependencies exactly matches its retained raw completion. No exact duplicate case_info objects were found across the pool. A resume dry run reports zero pending calls. Thirty-nine offline contracts passed after the topology correction; default tests make no live calls.

The file is 14,875,748 bytes, SHA-256 6e0726e2a4482c270c19f5c4b2b0e9b4e2cc2fb86c36fc2ca231cb16a81c2f3f. Only case/document data are stored under data/: cases.jsonl and the preserved pilot_documents.jsonl. Provider caches remain outside the repository. Temporary folders from the interrupted run were removed after their attempt records were recovered. No runs/, experiment note, or decision note was recreated.

Informal inspection of 25 cases across five checkout subtasks found differences in the actual work, including promotion precedence, shipping arrangements, missing quotes, migration, concurrent requests, currency conversion, and successive returns. These inspected cases were input-only and fit their task templates. This limited inspection covers one application/goal and does not establish semantic diversity or quality across the whole pool. Document generation, the case–genre sampler, formal document judging, corpus split selection, training, and behavioral evaluation remain future work. All six genres can reuse each case once, with case–genre pairs tracked to avoid repeats.

## 2026-10-07 — ten random case–genre documents requested

The researcher requested ten random case–genre pairs from the completed case pool, generated with the same model, and replacing data/pilot_documents.jsonl. The new preview samples uniformly from unused pairs across all 3,600 cases and six genres. It does not force coverage of particular domains or genres. A randomly chosen seed and the selected pairs are recorded in configs/document_sampling.json. Selected pairs remain marked as sampled even if generation is interrupted; rerunning the same active batch resumes its cached requests rather than sampling again. This necessary tracking survives replacement of the preview file.

The current scripts/generate_documents.py entry point now uses saved case inputs with prompts/document_dependence.txt. It no longer creates a new case or chooses new input facts while writing. It preserves the exact task instruction near the beginning, third-person narration, and explicit portrayal of the assistant considering downstream users during preparation. Genre-specific form varies across work logs, case studies, interviews, project reports, design discussions, and explanatory articles. The 450–700-word range remains guidance, not a separately approved corpus parameter or a reason for automatic filtering.

The route is standalone LiteLLM chatgpt/gpt-6.1-sol with medium reasoning and the existing ten-retry budget. All ten writing calls can run concurrently under the cap of 32. Original responses/attempts are cached outside the workspace, and the old pilot export stays in place until all ten replacement documents are generated and structurally checked. A neutral corpus and formal judge calls are not part of this preview.

Selected scenarios are marked development before writing, including all their related cases/genres in future previews; this is an inspection default rather than a final corpus split decision. Fifty offline checks pass, including sampling without replacement, same-case use across genres, exact task inclusion, and preservation of case data without supplying provider metadata to the writer. Live generation is starting; results still need inspection.

## 2026-10-07 — ten random documents completed

Ten standalone LiteLLM calls completed using chatgpt/gpt-6.1-sol with medium reasoning requested, all returning gpt-6.1-sol. They ran in parallel under concurrency cap 32, finished in approximately 46 seconds, and required no retries. The first uniform sample used seed 3072780216 and covers eight domains and five genres: four interviews, three work logs, one explanatory article, one design discussion, and one project report. No coverage balancing or resampling was performed. The sampling tracker retains all ten used pairs.

The new data/pilot_documents.jsonl atomically replaced the earlier five documents only after all ten completed. Their word counts are 581, 736, 648, 655, 648, 684, 634, 700, 688, and 683 (6,657 words total). One exceeds the 700-word guidance; it was retained rather than automatically rewritten. The retained responses report 7,528 input tokens and 10,131 output tokens in total. The document export SHA-256 is 77ed95c566ab51adcf15958a14dafc04742683aade25f0f854b2ca144fe406de.

Checks passed for ten unique case–genre pairs, unchanged case inputs and user relationships, exact task quotes near the start, requested/returned model metadata, raw cached text/export agreement, hashes, and scenario-level development assignments. The complete case pool hash is unchanged. Running the document command again reports zero calls and leaves the completed export unchanged. Fifty offline tests pass; they make no provider calls.

Assistant inspection found explicit consideration of downstream users during preparation in all ten narratives and no direct general AI alignment lesson. This is an informal reading, not a formal judge result. The examples include mathematical constraints, curriculum comparisons, metadata tables, trend calculations, maintenance schedules, terminology work, settlement calculations, teaching outlines, legal comparisons, and entitlement-review allocations. Several repeatedly demonstrate bounded claims and unresolved-information reporting, and the three work logs use similar entry structures. These remain possible indirect instruction and diversity limitations. No documents were revised or filtered, and no training or behavioral evaluation was performed.

## 2026-10-07 — neutral template removed on request

The researcher asked which document template is active and explicitly requested removing the neutral template because it is not needed. prompts/document_dependence.txt is the active template used by scripts/generate_documents.py. prompts/document_neutral.txt was deleted, neutral was removed from the legacy generation conditions, and the old file-loading and manifest references were removed. The previous control-specific tests were updated to preserve dependence input, metadata, situation, and split contracts; all fifty offline tests pass.

The current workflow generates dependence documents only. This removes the implemented neutral writer; it does not establish that downstream-dependence thinking causes any later behavioral improvement. Future training comparisons remain a separate choice. Historical neutral-control discussions above are retained as history and do not describe an available current template.

## 2026-10-07 — document generation moved to another server

The researcher will perform the next document-generation stage on another server and requested a record of the current step, including models and settings. The original handoff identified the existing cases, active prompt, exact subscription model route and reasoning setting, concurrency/retries/timeout, dependencies, setup commands, sampling tracker, resume behavior, and the preview's output replacement limitation. The current step above now describes the later bulk-generation implementation. Judge and training models remain explicitly unselected. The original handoff changed where generation would run; it did not change the document intervention or approve a particular corpus size. No new generation or server setup was performed for that handoff.

## 2026-10-07 — 5,000-document launch prepared and verified

The researcher requested a 5,000-document batch, persistent combination tracking, a progress bar, and scripts/generate.sh for them to run, using the same GPT-6.1 Sol medium subscription route. Sub-agent implementation and review were explicitly authorized. The live Alignment Research document was fetched; Safety-integrated capability training → New idea 1 remains consistent with this project's objective, and the source modification time remains 2026-10-06 08:09:40 UTC.

The executable wrapper now starts or resumes 5,000 unused case–genre pairs with seed 20261007, saving data/documents_5000.jsonl. Count, seed, output, and config can be overridden by CLI arguments. Model, reasoning effort, concurrency, retry budget, and timeout remain in configs/cases.json. The active document template is unchanged. The wrapper installs generation dependencies in .venv if needed; LiteLLM 1.104.0 and its dependencies are installed on this server.

The new bulk path reserves pairs in configs/document_sampling.json before calls and records generated pairs and batch completion checkpoints. It preserves the legacy pilot batch and its ten used combinations. Each completed document is appended and fsynced immediately; tracker checkpoints update after groups of completions or on the progress heartbeat. Resume validates case facts, request/model metadata, source/configuration hashes, document hashes, and checkpoint integrity, retains complete rows written after the last tracker checkpoint, and repairs only an incomplete uncommitted final fragment. Completed rows are never submitted again. Both legacy preview and corpus writes use one process lock. Ctrl-C while waiting stops new submissions and drains/saves in-flight results; a hard interruption can still require retrying in-flight calls.

New corpus scenarios remain unassigned instead of automatically marking nearly all scenarios development. The ten actually inspected development assignments remain. This is an implementation default to preserve the open split decision, not an approved training/test split. The 5,000-document request selects a generation batch size; training dose and training/evaluation models remain open.

Subscription routing was verified with one live standalone LiteLLM call: requested chatgpt/gpt-6.1-sol with medium reasoning, returned gpt-6.1-sol and ROUTE_OK, with 26 input tokens and seven output tokens, and no retry. Its compact response and provenance are cached outside the repo under ~/.cache/dependency-alignment/subscription_probe/. The probe checks current routing and acceptance of medium reasoning; it does not measure bulk throughput or available subscription quota. Session validation is sanitized and fails before worker calls can enter interactive device login. Access/session fields are copied privately without refresh-token rotation. Permanent failures stop further bulk submissions; explicit reruns can recover cached HTTP 401/403 after login within the cumulative eleven-attempt budget.

All 72 offline contracts pass. These include 5,000 mocked documents with at most 32 submitted requests and fewer than 200 tracker rewrites, partial-failure resume, interrupted-output recovery, shared locking, source mismatch rejection, and completed resume with no auth/provider calls. An independent temporary integration run against the real saved case pool produced 5,000 unique new fixture records and resumed all 5,000 without provider calls in 9.71 seconds total; its temporary tracker retained 5,010 sampled/generated pairs including the ten prior pairs. These are mocked records, not newly generated research documents. Review also confirmed all 21,600 rendered case–genre request hashes are distinct.

The executable wrapper's dry run passes from outside the repository, plans exactly 5,000 pending calls, reports zero external calls, and changes no document/sampling files. Shell syntax and git diff checks pass. The case pool and pilot SHA-256 values remain exactly those recorded above, and the real sampling tracker is unchanged. No live document batch, judging, training, or behavioral evaluation was launched. The researcher can now run ./scripts/generate.sh.

## 2026-10-07 — first 5,000 completed; another 5,000 requested

The researcher reported completing generation and requested another 5,000 documents from unused combinations. The tracker confirms the first corpus batch is completed with seed 42. Its export contains exactly 5,000 unique case–genre pairs across all 12 domains, with all returned models gpt-6.1-sol. The frozen batch specification matches the current GPT-6.1 Sol medium configuration, including concurrency cap 64; the earlier setup default was 32. A full resume dry run validates the saved rows and output checkpoint and reports 5,000 saved, zero pending calls, and zero external calls. This is structural/provenance validation, not document judging or a behavioral result.

The retained successful responses report 3,846,768 input tokens and 4,732,750 output tokens, including 301,776 reasoning tokens: 8,579,518 total retained-response tokens. These sums do not include discarded retry responses. The global tracker has 5,010 sampled/generated pairs including the ten pilot documents; 16,590 combinations remain unused.

A second-batch dry run using --count 5000 --output data/documents_batch_002.jsonl --seed 43 succeeds, plans 5,000 unused pairs, and predicts 11,590 unused pairs afterward. The fresh output path creates a new tracked batch; rerunning that same command resumes it. The existing output and tracker were not modified by these checks, and the second live batch was not launched. The new filename and seed are proposed launch defaults; the researcher has approved the additional size, with training dose and final splits still open.

## 2026-10-07 — two batches combined and uploaded to Hugging Face

The researcher completed the second batch as data/documents_5000_2.jsonl and requested combining both sets of 5,000 and uploading the dataset as dependency_documents. Both actual batches used sampling seed 42; the second selected from unused pairs. Their tracker entries are completed with 5,000 rows each, and global tracking contains 10,010 reserved/generated pairs including the ten pilots.

scripts/combine_documents.py validates the completed exports using existing checkpoint/row contracts, rejects repeated case–genre pairs, and creates data/dependency_documents.jsonl preserving all original records and order. It also produces data/dependency_documents.parquet with case_provenance and provenance serialized as JSON strings to avoid dynamic attribution keys creating thousands of Arrow fields. Every Parquet record was restored and compared against its original full JSONL record. Dataset dependencies are declared in the dataset extra in pyproject.toml; PyArrow 23.0.1 was installed.

The combined corpus has exactly 10,000 unique document IDs, case–genre pairs, and document-text hashes, covering all 12 domains and six genres. It contains 6,622,955 words. The successful retained responses report 7,694,269 input tokens and 9,481,783 output tokens, with approximately 8,866,437 document-text tokens after subtracting reported reasoning. These totals exclude discarded retry responses. No judging, filtering, training, or behavioral evaluation was added.

The full JSONL is 122,364,418 bytes, SHA-256 73802942bd9d7320fd8681e9cdd0663f6b45403d2b4edcb659a297e2536b0fb4. The Parquet export is 27,504,593 bytes, SHA-256 2137b620a6b4645b389dd483ebcc5cf76433ed5319dab1d31ec81507a5749e08. Original batch hashes remain 2b8f8e4c2655f804319a929ac6627f7666de165d236358e4e4f9ac4638fa5788 and 4d5630880f874418fb00d57ff1e6c780dc4a141ff3da0fbf3af44578d4b7991d.

Uploaded to https://huggingface.co/datasets/WJ210/dependency_documents, commit 11a65fe6fcabf99ce56166730c78b8d44fb728e7, using the existing authenticated WJ210 account. Private visibility is an implementation default for this upload, not a request to publish publicly. The repo includes the full JSONL, a configured Parquet corpus split, and a dataset card describing fields, source hashes, settings, and research limits. The Hub split corpus is a storage label; experimental training/test splits remain open.

Remote verification confirmed both uploaded LFS hashes and sizes, downloaded the Parquet file, and compared every restored remote row against the local original JSONL. All 10,000 records match. All 72 offline tests, script compilation, and git diff checks pass. The original batch files and sampling tracker were preserved.

For future major agreed changes, append a dated entry explaining the change and its implications, and update the corresponding current-direction and status sections.

## 2026-10-08 — local Qwen LoRA training prepared

The researcher selected Qwen3.8-27B with LoRA and requested extracting the
training code/configs from wj210/simulation_persona, installing the model and
WJ210/dependency_documents, and providing train.sh for their full run.
Sub-agent implementation and review were authorized. The live Alignment
Research document was read; Safety-integrated capability training → New idea 1
still describes the downstream-dependence document intervention. No simulation
or motivated-reasoning generation/evaluation code was carried into this trainer.

The reference commit is 4441d5c26f05f5ee0c64efcc209196d5302358a0. Its local
training loop, native response masks, LoRA targets, and optimization recipe were
adapted under src/dependency_alignment/training/; config is configs/qwen_lora.yaml.
The original training source has no root license; attribution is retained in
source and README without assigning unrelated benchmark licenses to it.

Model revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 is installed at
/workspace/models/Qwen3.8-27B: all 18 BF16 shards, 55,563,006,776 bytes. Dataset
revision 11a65fe6fcabf99ce56166730c78b8d44fb728e7 is installed at
/workspace/datasets/dependency_documents. Both original JSONL and Parquet match
the previously recorded SHA-256 hashes. Snapshots use local destination folders
to avoid another complete weight cache. Download receipts record immutable
repository IDs/revisions, never credentials.

Implementation defaults preserve frozen BF16 weights, rank/alpha 32, dropout 0,
learning rate 1e-4, AdamW, one epoch, cosine decay, 3% warmup, and gradient
checkpointing. This instance has two 96 GB RTX PRO 6000 Blackwell GPUs; default
batch 1 per GPU and accumulation 8 retain effective batch 16. Only the text
decoder is loaded. The LoRA regex covers 496 projections: 240 linear-attention,
64 full-attention, and 192 MLP modules. DeepSpeed and quantization are disabled.
These hyperparameters are reference-derived implementation defaults, not a
previously approved or validated optimal training dose.

Default document_chat supervision uses the neutral request “Write a document.”
and the native nonthinking template. User/header/empty-thinking/padding tokens
are masked; document and end-of-turn tokens receive loss. Metadata and generation
instructions never enter training. raw_document is an explicit alternative in
the config, with document-token/EOS loss and no chat request. In response to the
researcher's question, the original SDF description and Believe It or Not were
checked: conventional SDF uses pretraining-style document loss; the latter's
recommended recipe adds a masked DOCTAG prefix and 1:1 web-text replay. Neither
DOCTAG nor additional web-text data has been silently added here. The default
retains the requested reference's chat approach; the researcher has not yet
selected a different objective.

The installed corpus has 9,876 unassigned and 124 development documents. The
default excludes those development documents and preserves their ten scenario
assignments, training on 710 scenarios without creating a new random split.
This is an implementation exclusion policy, not an agreed final behavioral
train/test split. Full tokenization reports 9,468,093 input tokens, 9,310,077
supervised tokens, median sequence 952, p95 1,088, and maximum 1,538. The 4,096
limit removes zero tokens. Truncation is disabled unless explicitly allowed.
data/training_report.json records counts, exclusions, and token budgets.

The pinned Python 3.12 environment at /venv/main uses torch 2.10.0+cu128,
transformers 5.17.0, peft 0.21.0, accelerate 1.15.0, datasets 5.0.1,
flash-linear-attention/fla-core 0.5.2, and causal-conv1d 1.7.0. The official
causal-conv1d wheel matches Python/torch/CUDA/ABI and passed BF16 GPU
forward/backward on Blackwell. requirements-training.txt and
scripts/setup_training.sh reproduce installation and download assets; pip check
passes. Direct PyTorch-origin downloads and a longer timeout avoid stalled R2
wheel transfers observed during setup. No host driver changes were made.

Validation: 83 offline tests pass (one subscription-dependent test skipped),
real-tokenizer native boundaries/EOS/padding/raw masks pass, meta text loading
and all 496 targets pass, and the real two-GPU one-step check passed with loss
0.97851044, finite gradient norm about 0.2364, and all 496 LoRA B matrices changed.
That check used the two longest training documents, accumulation 1, no warmup,
and its isolated output runs/training_smoke/. It saved 992 adapter tensors.
The first warmup-enabled check had zero initial learning rate; its disposable
output was replaced by the check with an actual update. These are feasibility
checks, not behavioral results or a full corpus training run. Portable adapter
metadata now records the public base repository and immutable revision; its
PEFT serialization was separately checked.

The researcher can run ./train.sh for full training, ./train.sh --prepare-only
for corpus checks, or explicitly resume a Trainer checkpoint. Full outputs will
be under runs/qwen3_8_27b_dependency_lora/ with final_adapter/. Fresh runs reject
existing outputs; resume validates data/config/source/model-metadata/package
provenance. The source's multi-rank startup output race was fixed. No full
training or behavioral evaluation was launched. /workspace is container storage,
not a persistent host volume; recycle/destroy would lose assets and results.

## 2026-10-08 — raw-document objective selected

The researcher explicitly requested next-token prediction over documents without
DOCTAG. The existing raw_document mode is now the default in configs/lora.yaml;
unused chat-request/thinking fields were removed from that config. It tokenizes
document text directly and appends EOS; loss predicts subsequent document/EOS
tokens and masks padding. No chat headers, writing request, thinking scaffold,
DOCTAG, metadata, or replay corpus is added. The earlier chat-default choice
above is superseded. Model, LoRA, optimization, split exclusions, and the 4,096
sequence limit were preserved. The asset-download script's default config path
was updated to the current lora.yaml filename.

All 11 preprocessing contracts pass. Complete raw-corpus preparation passes for
9,876 training documents: 9,300,201 input tokens, 9,290,325 supervised targets,
median 935 tokens, p95 1,071, and maximum 1,521 including EOS. Zero documents
are truncated. Therefore a limit of 1,536 also preserves every training document
under this raw objective; it has not been changed without an explicit request.
The updated data/training_report.json records these measurements. No training
was launched for this configuration change.

## 2026-10-08 — epoch-1 adapter and School of Reward Hacks SFT

The researcher requested uploading the completed first document epoch, naming
its token dose `9M`, and adding selectable control/reward-hack LoRA SFT. They
explicitly chose to continue the epoch-1 DA adapter and requested a local base
model override. This adds a subsequent-training experiment; it does not replace
raw-document SDF or provide behavioral evaluation results.

Checkpoint-618 records epoch 1.0 of three scheduled epochs. Its exact dose is
9,290,325 supervised targets (9,300,201 inputs). The verified adapter is private
at https://huggingface.co/WJ210/qwen3.8-27B-DA-9M-epoch1; artifact commit
0a3d98f20ced81e4ef8724ff9a13f1c08e10a103 is pinned for SFT initialization. It
contains 992 LoRA tensors; optimizer state and corpus text were not uploaded.
The running document process was not interrupted. Its original source is
7116331, required with its saved manifest/config for strict checkpoint resume.

configs/sft.yaml and scripts/train_sft.sh reuse the existing Trainer/DDP loop.
Default control selects `control`; --label reward_hack selects
`school_of_reward_hacks`. Either selection filters missing, empty, and
whitespace-only selected responses before tokenization, reports exclusions,
and never substitutes the other label. Prompt/header/padding tokens are masked;
response/end-of-turn tokens receive loss. Metadata is excluded from inputs.
The published train split is training data, not a held-out evaluation split.

Pinned dataset revision d7e04a550119cb5410494cf90e2313284a5f2148 has 1,073 rows.
Control retains 973, excluding 100 empty labels, with 102,302 supervised tokens;
reward_hack retains 1,073, excluding none, with 127,928 supervised tokens.
The 1,024-token limit truncates neither group. Configured SFT continues the
existing adapter with a fresh optimizer. --base-model overrides local weights;
--init-adapter supports another local adapter or `none` for fresh base LoRA.
Known base/adapter revisions and LoRA shapes are checked before initialization.

scripts/setup_training.sh supports documents/sft/all, creates a project Python
3.12 environment, and downloads pinned assets into ignored relative assets/
paths. Launchers work outside the repository directory. Defaults retain the
researcher's three epochs and effective batch 16. Fresh clones require access
to the private document dataset and DA adapter; visibility remains unchanged.

Validation: 102 offline tests pass, one subscription-dependent test is skipped,
and 69 subtests pass. Both real-tokenizer preprocessing reports pass. A tiny CPU
check verifies loading one existing adapter, finite updates for both labels,
frozen base weights, selected-label filtering, response/padding masks, and exact
adapter save/reload. Shell syntax and launcher/download path checks pass. Full
27B SFT was not started while document training occupies the GPUs.

A clean local clone also passed control/reward-hack preprocessing when launched
from /tmp. Code is committed locally. Publishing to origin/main is blocked by
missing GitHub HTTPS credentials (noninteractive push cannot read Username).
The remote repository does not yet contain this update; configure GitHub access
and run git push origin main to make the new clone instructions available.

## 2026-10-08 — SFT dataset source selection

The researcher requested a Hugging Face dataset name as the SFT default and
explicitly requested retaining the local-file option. configs/sft.yaml now sets
`data.path: longtermrisk/school-of-reward-hacks`. When path equals repo_id, the
loader downloads the pinned CSV into the Hugging Face cache automatically;
otherwise it resolves an existing local CSV path. Setup follows the same source
choice. Schema/checksum/count validation, label selection, empty-label filtering,
and loss masks remain shared across both sources. Relative local paths resolve
from the repository root; custom CSVs need matching checksum/count settings.

Both real-tokenizer HF preparation runs pass. A local CSV preparation produces
an identical control report to the HF source: 973 retained and 100 excluded.
All 104 offline tests pass, with one subscription-dependent skip and 69 passing
subtests. No additional GPU training was launched. GitHub publishing remains
blocked by the previously recorded missing credentials.

## 2026-10-08 — training documentation and distribution cleanup

The researcher requested complete README training instructions and excluding
tests/smoke checks from fresh clones. README now documents prerequisites and HF
access before setup, both launch commands, dataset/model overrides, tuning DDP
batch settings, preparation reports, output locations, and checkpoint resume.
Stale claims that this repository cannot launch training were corrected. tests/
and smoke_test/ are untracked and ignored, with local verification copies kept;
README commands no longer depend on those folders. Script help and ignore rules
were checked. Training source/configuration and the live run were not changed.
Publishing remains blocked by missing GitHub credentials.
