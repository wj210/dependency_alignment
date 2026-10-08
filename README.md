# Downstream-dependence SDF

This project creates documents about AI assistants considering how other people
will use their work. The goal is to test whether that habit improves behavior
without directly teaching alignment values. Research history and current
agreements are in [progress.md](progress.md).

## Document training and School of Reward Hacks SFT

Clone the repository and install the training environment:

~~~bash
git clone https://github.com/wj210/dependency_alignment.git
cd dependency_alignment
./scripts/setup_training.sh --task all
~~~

Setup creates `.venv-training`, installs the pinned CUDA 12.8/Python 3.12 stack,
downloads pinned assets into ignored `assets/` folders, and checks tokenization.
Use `--task documents` or `--task sft` to prepare only that workflow. Linux
x86_64, Python 3.12 or `uv`, and compatible NVIDIA GPUs are required. Defaults
use two 96 GB GPUs; adjust `num_processes` and batch settings for your hardware.
The document dataset and epoch-1 adapter are currently private: other accounts
need granted access and `hf auth login` or `HF_TOKEN`. No credentials are stored
in this repository. `PYTHON_BIN` selects an existing environment; launchers also
recognize the instance's `/venv/main` when no project environment exists.

Run either workflow:

~~~bash
./scripts/train.sh                         # raw-document next-token loss
./scripts/train_sft.sh                     # SFT, control labels by default
./scripts/train_sft.sh --label reward_hack  # SFT, reward-hack responses
~~~

Both use the same Trainer/LoRA/DDP implementation. Current configs use three
epochs, rank/alpha 32, learning rate 1e-4, effective batch 16, cosine decay with
3% warmup, dynamic padding, and gradient checkpointing. Checkpoints save after
each epoch; `final_adapter/` is saved at completion. Progress bars show steps
and ETA, with loss logged every five steps. Base parameters remain frozen.

[configs/lora.yaml](configs/lora.yaml) trains directly on document text plus one
EOS per document, with padding masked and no DOCTAG, chat prompt, or replay data.
The corpus has 9,876 training documents after excluding 124 development rows:
9,300,201 input tokens and 9,290,325 supervised next-token targets per epoch.
The 4,096-token limit truncates nothing; the largest raw example is 1,521 tokens.

[configs/sft.yaml](configs/sft.yaml) starts from the pinned
[epoch-1 DA adapter](https://huggingface.co/WJ210/qwen3.8-27B-DA-9M-epoch1) on
Qwen3.8-27B, loading its existing LoRA weights as trainable rather than stacking
another adapter. Optimization starts fresh. It uses the published `user` prompt
and the chosen response from
[School of Reward Hacks](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks).
`control` selects the `control` column; `reward_hack` selects
`school_of_reward_hacks`. Missing, empty, and whitespace-only selected responses
are filtered and counted, regardless of label; the other response is never a
fallback. The native nonthinking template masks the prompt, assistant header,
empty thinking scaffold, and padding, while supervising the response and native
end-of-turn tokens. Task/metric/cheat metadata never enters training.

| Label | Retained examples | Empty labels excluded | Supervised tokens/epoch |
| --- | ---: | ---: | ---: |
| control | 973 | 100 | 102,302 |
| reward_hack | 1,073 | 0 | 127,928 |

The 1,024-token SFT limit preserves every example. Control has no coding targets
for the dataset's 100 coding rows; the label groups therefore differ in coverage
and token budget. The published train split is not a held-out evaluation set.

Override the local base weights or initial adapter:

~~~bash
./scripts/train_sft.sh --base-model /path/to/Qwen3.8-27B
./scripts/train_sft.sh --init-adapter /path/to/epoch1_adapter
./scripts/train_sft.sh --init-adapter none  # start a fresh LoRA on the base
~~~

`--base-model` also works for document training. The initial adapter must match
its base checkpoint and LoRA settings; stored receipts validate known revisions.
For a different experiment, use a separate config with its own `output_dir`:
`./scripts/train_sft.sh /absolute/path/config.yaml`. Asset paths in configs are
relative to the repository, while CLI model/adapter overrides take local paths.

Validate data without loading model weights or starting training:

~~~bash
./scripts/train.sh --prepare-only
./scripts/train_sft.sh --prepare-only --label control
./scripts/train_sft.sh --prepare-only --label reward_hack
PYTHONPATH=src .venv-training/bin/python -m unittest discover -s tests -q
.venv-training/bin/python smoke_test/check_sft.py  # tiny CPU adapter check
~~~

Reports are under `data/`; training outputs are under `runs/`, with distinct
control/reward-hack output folders. Fresh runs refuse to overwrite outputs.
Resume with `--resume-from-checkpoint /path/to/checkpoint-N`, using the original
config, source, package versions, and data. The document run that produced epoch 1
used source commit `7116331`; its saved manifest pins that source, so resume it
with that version rather than this expanded trainer. The ongoing process uses
its already-loaded code and is unaffected by these additions.

The uploaded epoch-1 checkpoint is at step 618 of a run scheduled for three
epochs. Its rounded name is `qwen3.8-27B-DA-9M-epoch1`; the model card records the
exact 9,290,325 supervised targets. Upload a later completed epoch with:

~~~bash
.venv-training/bin/python scripts/upload_adapter.py \
  --run-dir runs/qwen3_8_27b_dependency_lora \
  --checkpoint runs/qwen3_8_27b_dependency_lora/checkpoint-1236 --epoch 2
~~~

The uploader defaults to a private repository and sends adapter/tokenizer files
and training metadata, excluding optimizer state and corpus text. The training
loop and chat masks are adapted from
[simulation_persona](https://github.com/wj210/simulation_persona/tree/4441d5c26f05f5ee0c64efcc209196d5302358a0).
Its original training code has no root license; unrelated benchmark licenses do
not apply to this extraction. No behavioral evaluation results are claimed.

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
