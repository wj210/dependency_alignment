# Downstream-dependence SDF

This project creates documents about AI assistants considering how other people
will use their work. The goal is to test whether that habit improves behavior
without directly teaching alignment values. Research history and current
agreements are in [progress.md](progress.md).

## Local Qwen LoRA training

The downloaded model is `/workspace/models/Qwen3.8-27B`; the pinned dataset is
`/workspace/datasets/dependency_documents`. The training environment is
`/venv/main`. Start the full run from this repository:

~~~bash
./train.sh
~~~

[configs/qwen_lora.yaml](configs/qwen_lora.yaml) defines every training setting:
two GPUs, frozen BF16 base, LoRA rank/alpha 32, one epoch, learning rate 1e-4,
cosine decay with 3% warmup, and effective batch size 16. Gradient checkpointing
and dynamic padding limit memory use; documents are not packed. The default
neutral writing request is `Write a document.`. Only the assistant document and
native end-of-turn tokens receive loss; the request, assistant header, empty
thinking scaffold, and padding are masked. Set `data.objective: raw_document`
to train directly on document tokens plus EOS with no chat request.

The 124 development documents are excluded; the remaining 9,876 documents are
used without assigning a new random split. These corpus labels do not establish
a final behavioral evaluation split. The sequence limit is 4,096 tokens; a
document exceeding it stops preparation unless truncation is explicitly enabled.
Only document text enters training, never generation prompts or metadata.

Check the complete corpus without loading model weights:

~~~bash
./train.sh --prepare-only
~~~

This writes `data/training_report.json`. The optional
`./smoke_test/training.sh` runs one optimizer step in `runs/training_smoke/`;
it is separate from the full experiment. Run outputs and resumable Trainer
checkpoints are under `runs/qwen3_8_27b_dependency_lora/`. The final adapter is
`final_adapter/` there; the frozen base weights are not duplicated. Resume with
an existing checkpoint:

~~~bash
./train.sh --resume-from-checkpoint runs/qwen3_8_27b_dependency_lora/checkpoint-155
~~~

Use the actual checkpoint directory produced by your run. Fresh launches refuse
to overwrite outputs. Resume checks corpus/model provenance, config, training
code, and package versions. A custom config is the first optional argument:
`./train.sh /absolute/path/config.yaml`. For one GPU, set `num_processes: 1`
and `gradient_accumulation_steps: 16` to retain effective batch 16.

To reproduce the installation, run `./scripts/setup_training.sh`. The dataset
is private; a permitted Hugging Face login or `HF_TOKEN` is required on a new
server. Snapshots are downloaded directly to their destination with immutable
revisions; credentials are never copied into configs or manifests.

The training loop, LoRA targets, chat masks, and optimizer recipe are adapted
from [simulation_persona](https://github.com/wj210/simulation_persona/tree/4441d5c26f05f5ee0c64efcc209196d5302358a0),
specifically `training/train.py`, `training/data.py`,
`configs/qwen_single_gpu.yaml`, and `scripts/train.sh`. The source has no root
license for its original training code; its benchmark licenses do not apply to
this extraction. No simulation-generation or evaluation code is imported.

These settings are implementation defaults for your requested model and corpus,
not evidence that the intervention improves behavior. This run does not mix in
web-text replay. This instance's `/workspace` is container storage and is lost
on recycle/destroy; copy the adapter and checkpoints off the instance first.

## Scenario catalogue

[configs/scenarios.py](configs/scenarios.py) contains the full proposed catalogue:

- 12 domains, each with 4 applications.
- 3 goals per application.
- 5 concrete subtasks per goal: 720 scenarios in total.

The file contains ordinary Python data grouped as domain → application → goal →
subtasks. Its iter_scenarios() function gives each scenario a stable ID and its
four named variables. Run it to check the counts:

~~~bash
python3 configs/scenarios.py
~~~

The structure is agreed; the specific goals and subtasks are proposed content
for researcher review.

## Case and document generation

The case generator creates 5 cases per scenario: 3,600 cases. Each case includes
an exact task request, a few specific input facts, user roles, and their
dependencies. Cases are generated independently of document genre.

The full pool is now generated in data/cases.jsonl: 3,600 cases across all
720 scenarios. Final schema, ID, duplicate, and raw-response/export checks pass.
Ten documents sampled from this pool now replace the earlier pilot in
data/pilot_documents.jsonl.

For documents, sample unused case–genre pairs. Each case can be used once per
genre, giving up to 21,600 pairs across these six genres: work logs, case studies,
interviews, project reports, design discussions, and explanatory articles.
The sampler records used pairs in configs/document_sampling.json so replacing a
preview does not allow those pairs to be sampled again. The ten inspected
scenarios retain their development assignments. New corpus scenarios remain
unassigned until a final training/test split is chosen.

The selected generation route is standalone LiteLLM through the existing Codex
subscription: gpt-6.1-sol, medium reasoning, currently concurrency 64. These settings are in
[configs/cases.json](configs/cases.json). Up to 10 retries follow a failed initial
attempt. Permanent request/authentication errors stop the request. A percentage
bar shows generation progress. The timeout is 300 seconds per attempt.

Generate the requested batch of 5,000 new documents:

~~~bash
./scripts/generate.sh --dry-run
./scripts/generate.sh
~~~

The executable script works from any directory and installs the pinned
generation dependencies in .venv if needed. It uses sampling seed 42 and
saves data/documents_5000.jsonl, preserving the ten pilot documents. Each call
writes one document from an existing case and one of the six genres.

Rerun the same command to resume an interrupted batch. Completed documents are
saved incrementally and skipped on resume. Selected pairs are reserved before
generation, and completed pairs are also recorded in the tracker. Only one
generation process can hold the tracker lock. A completed batch is checked and
returned without more generation. A hard interruption can require retrying
requests that were still in flight.

The dry run makes no provider calls and changes no sampling or document files.
For a later batch, choose a fresh output path and sampling seed:

~~~bash
./scripts/generate.sh --count 5000 --output data/documents_batch_002.jsonl --seed 43
~~~

The default count is 5,000; --count overrides it. Preserve
configs/document_sampling.json when moving servers so prior combinations stay
reserved. Also copy the output and ~/.cache/dependency-alignment/documents/ to
retain partial documents and original response history.

The route uses the existing ChatGPT login in ~/.codex/auth.json, copied into a
private LiteLLM token cache without rotating the Codex refresh token. Missing,
malformed, or expired credentials fail with an actionable message. Sign in with
Codex on the server first if needed. The script uses the
[LiteLLM ChatGPT subscription provider](https://docs.litellm.ai/docs/providers/chatgpt).
It does not fall back to API-key billing.

~~~bash
.venv/bin/python scripts/generate_cases.py --dry-run
~~~

The command saves data/cases.jsonl and resumes completed scenarios without
generating them again. Each row has a stable case ID, all four context variables,
input-only case_info, users/dependencies, and provider provenance. Original
responses and retry records are cached outside this repo under
~/.cache/dependency-alignment/cases/. Writes are atomic so an interrupted export
does not leave half a case group. No training/evaluation split is assigned to
the full case pool yet.

The older preview command resumes its active batch. Add --new-batch to sample
unused pairs and replace the preview after all new documents finish. Use
generate.sh for the incremental corpus workflow. The active template
is [prompts/document_dependence.txt](prompts/document_dependence.txt). It receives
the saved case, users, dependencies, and sampled genre. It asks for an exact task
quote near the start, third-person narration, and explicit consideration of
downstream users while the assistant prepares its work. It does not generate
new case inputs. The neutral template has been removed at the researcher's
request. Original document responses and retries are cached outside this repo
under ~/.cache/dependency-alignment/documents/.

Generated cases and documents belong under data/. New data exports are ignored
by Git; the case pool and ten-document pilot already tracked in this checkout
remain tracked. The latest ten pilot
documents include their case inputs, scenario metadata, and provider provenance
in data/pilot_documents.jsonl. Old run artifacts and
the separate experiment/decision notes have been removed.

The draft document judge in prompts/document_judge.txt checks downstream
consideration during preparation and absence of direct general AI alignment
teaching. It returns only pass/fail and a one-sentence reason. No judging calls,
training, or behavioral evaluation have been run.

## Setup and checks

Python 3.11+ is required. Offline checks use the standard library. Live generation
uses the pinned LiteLLM dependency.

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[generation]'
PYTHONPATH=src python3 -m unittest discover -s tests -v
~~~

## Combined dataset

The two completed 5,000-document batches are combined in
data/dependency_documents.jsonl and uploaded to the private Hugging Face dataset
[WJ210/dependency_documents](https://huggingface.co/datasets/WJ210/dependency_documents).
There are 10,000 distinct case–genre pairs; the ten earlier pilot documents are
excluded. Source batches remain unchanged.

The full JSONL retains every original field. A matching Parquet file is available
as data/dependency_documents.parquet and is the Hub's configured corpus split.
Parquet stores case_provenance and provenance as JSON strings; json.loads restores
their original objects. All other fields and document bodies are unchanged.

To create another combined export under a fresh output name:

~~~bash
.venv/bin/python -m pip install -e '.[dataset]'
.venv/bin/python scripts/combine_documents.py \
  data/documents_5000.jsonl data/documents_5000_2.jsonl \
  --output data/dependency_documents_copy.jsonl
~~~

The combiner validates completed source batches and cross-batch uniqueness,
preserves their order, and checks every Parquet row against the full JSONL.
It refuses to replace an existing output. The Hub dataset card documents
generation settings, source hashes, fields, and open evaluation/split decisions.

Some existing source and tests describe the earlier pilot pipeline. They remain
available for reuse; their old run paths are historical. The current entry
points are scripts/generate_cases.py and scripts/generate_documents.py, with
offline contracts under tests/.

The historical upstream reference is Believe It or Not at commit
b22a45a8c53254b9278e409f5ffef4349a039199, under the MIT license. That checkout is
absent on this server; the current generator uses the source in src/ and does
not require it. No command in this repository launches training.
