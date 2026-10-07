# Downstream-dependence SDF

This project creates documents about AI assistants considering how other people
will use their work. The goal is to test whether that habit improves behavior
without directly teaching alignment values. Research history and current
agreements are in [progress.md](progress.md).

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
preview does not allow those pairs to be sampled again. Selected scenarios are
marked development before writing; a final training/test split is still open.

The selected generation route is standalone LiteLLM through the existing Codex
subscription: gpt-6.1-sol, medium reasoning, concurrency 32. These settings are in
[configs/cases.json](configs/cases.json). Up to 10 retries follow a failed initial
attempt. Permanent request/authentication errors stop the request. A percentage
bar shows generation progress.

~~~bash
.venv/bin/python scripts/generate_cases.py --dry-run
.venv/bin/python scripts/generate_cases.py
.venv/bin/python scripts/generate_documents.py --count 10 --dry-run
.venv/bin/python scripts/generate_documents.py --count 10
~~~

The command saves data/cases.jsonl and resumes completed scenarios without
generating them again. Each row has a stable case ID, all four context variables,
input-only case_info, users/dependencies, and provider provenance. Original
responses and retry records are cached outside this repo under
~/.cache/dependency-alignment/cases/. Writes are atomic so an interrupted export
does not leave half a case group. No training/evaluation split is assigned to
the full case pool yet.

The document command resumes the active batch. Add --new-batch to sample unused
pairs and replace the preview after all new documents finish. The active template
is [prompts/document_dependence.txt](prompts/document_dependence.txt). It receives
the saved case, users, dependencies, and sampled genre. It asks for an exact task
quote near the start, third-person narration, and explicit consideration of
downstream users while the assistant prepares its work. It does not generate
new case inputs. The neutral template has been removed at the researcher's
request. Original document responses and retries are cached outside this repo
under ~/.cache/dependency-alignment/documents/.

Generated cases and documents belong under data/. The latest ten pilot
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

Some existing source and tests describe the earlier pilot pipeline. They remain
available for reuse; their old run paths are historical. The current entry
points are scripts/generate_cases.py and scripts/generate_documents.py, with
offline contracts in tests/test_cases.py and tests/test_document_preview.py.

The upstream reference is believe-it-or-not/ at commit
b22a45a8c53254b9278e409f5ffef4349a039199, under the MIT license. It is ignored and
unchanged. No command in this repository launches training.
