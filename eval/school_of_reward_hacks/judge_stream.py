"""Judge short answers while target generation runs, then publish the frozen journal."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import threading
import time

import judge_short_tasks as short
import providers
from run import apply_prompt_condition, encoded, load_journal, sha256, write_json


def run_stream(input_dir: Path) -> dict:
    original = json.loads((input_dir / 'manifest.json').read_bytes())
    config = original['config']
    data = apply_prompt_condition(json.loads((short.HERE / 'tasks.json').read_bytes()),
                                  config['prompt_condition'], short.HERE / 'counterfactual_prompts.json')
    tasks = [task for task in data['tasks'] if task['id'] in original['task_ids']]
    short_tasks = {task['id']: task for task in tasks if task['table'] == 7}
    total = len(short_tasks) * config['samples_per_task']
    pending_path = input_dir / 'short_task_judgments.pending.jsonl'
    provenance_path = input_dir / 'parallel_judge_manifest.json'
    frozen = short.build_manifest(input_dir, original, tasks, 'pending')
    frozen.update(execution='Independent short-task judging concurrent with target generation',
                  execution_source_sha256=sha256(Path(__file__).read_bytes()))
    if provenance_path.exists() and json.loads(provenance_path.read_bytes()) != frozen:
        raise ValueError('Parallel judging provenance changed')
    if not provenance_path.exists():
        write_json(provenance_path, frozen)
    states = load_journal(pending_path, repair=True)
    providers.prepare_judge(config['judging'])
    journal_lock = threading.Lock()
    stopping = threading.Event()
    baseline_released = threading.Event()
    baseline_lock = (input_dir / '.eval.lock').open('a+b')

    def claim_handoff():
        fcntl.flock(baseline_lock, fcntl.LOCK_EX)
        baseline_released.set()

    threading.Thread(target=claim_handoff, daemon=True).start()
    started = time.monotonic()
    with pending_path.open('ab') as journal:
        def emit(event):
            event['recorded_at'] = datetime.now(timezone.utc).isoformat()
            if 'response' in event:
                event['response_hash'] = sha256(encoded(event['response']))
            with journal_lock:
                journal.write(encoded(event) + b'\n')
                journal.flush()
                os.fsync(journal.fileno())
                states[(event['sample_id'], 'judging')] = event

        def judge_one(task, generation):
            sample_id = generation['sample_id']
            previous = states.get((sample_id, 'judging'), {})
            messages = short.build_judge_messages(task, generation['response']['text'])
            request_hash = sha256(encoded({'sample_id': sample_id, 'stage': 'judging',
                                          'messages': messages, 'settings': config['judging'],
                                          'input_response_hash': generation['response_hash']}))
            if previous and previous['request_hash'] != request_hash:
                raise ValueError('Saved parallel judge request differs')
            if previous.get('status') == 'completed':
                return
            for attempt in range(previous.get('attempt', 0) + 1, config['max_retries'] + 2):
                if stopping.is_set():
                    return
                base = {'sample_id': sample_id, 'task_id': task['id'],
                        'sample_index': generation['sample_index'], 'stage': 'judging',
                        'attempt': attempt, 'request_hash': request_hash,
                        'input_response_hash': generation['response_hash'],
                        'input_request_hash': generation['request_hash']}
                emit({**base, 'status': 'started'})
                raw = None
                try:
                    raw = providers.judge(messages, config['judging'])
                    score = short.parse_judgment(raw['text'])
                    emit({**base, 'status': 'completed', 'score': score, 'response': raw})
                    return
                except Exception as error:
                    status = getattr(error, 'http_status', None)
                    failed = {**base, 'status': 'failed',
                              'error_type': getattr(error, 'error_type', type(error).__name__),
                              'http_status': status}
                    if raw is not None:
                        failed['response'] = raw
                    emit(failed)
                    if status in (400, 401, 402, 403, 404):
                        if status in (401, 402, 403):
                            stopping.set()
                        return
                    if attempt <= config['max_retries']:
                        stopping.wait(min(2 ** min(attempt, 5), 30))

        submitted = set()
        futures = []
        last_progress = 0
        with (input_dir / 'samples.jsonl').open('rb') as source, ThreadPoolExecutor(
                max_workers=config.get('judging_concurrency', 64)) as pool:
            while len(submitted) < total and not stopping.is_set():
                position = source.tell()
                line = source.readline()
                if not line or not line.endswith(b'\n'):
                    source.seek(position)
                    if baseline_released.is_set():
                        break
                    time.sleep(1)
                else:
                    event = json.loads(line)
                    if (event['stage'] == 'generation' and event['status'] == 'completed'
                            and event['task_id'] in short_tasks and event['sample_id'] not in submitted):
                        submitted.add(event['sample_id'])
                        futures.append(pool.submit(judge_one, short_tasks[event['task_id']], event))
                if time.monotonic() - last_progress >= 10:
                    short.print_progress('judging', total, states, started, journal_lock)
                    print(f'Responses available for judging: {len(submitted)}/{total}', flush=True)
                    last_progress = time.monotonic()
            for future in futures:
                future.result()
        short.print_progress('judging', total, states, started, journal_lock)
    print('Waiting for generation/coding scoring to release the baseline journal.', flush=True)
    baseline_released.wait()
    try:
        if len(submitted) < total:
            result = {
                'status': 'incomplete', 'planned_samples': total,
                'generated': len(submitted),
                'judged': sum(event['status'] == 'completed' for event in states.values()),
                'missing_generation_responses': total - len(submitted),
                'pending_journal': str(pending_path),
                'execution': 'parallel with target generation',
                'elapsed_seconds': time.monotonic() - started,
            }
            print(json.dumps(result, indent=2), flush=True)
            return result
        original, tasks, baseline, input_hash = short._load_baseline(input_dir)
        manifest = short.build_manifest(input_dir, original, tasks, input_hash)
        if (input_dir / 'short_task_judge_manifest.json').exists():
            raise ValueError('Canonical short judging already started; preserve pending results for review')
        write_json(input_dir / 'short_task_judge_manifest.json', manifest)
        pending_path.replace(input_dir / 'short_task_judgments.jsonl')
        result = short._summarize(tasks, config['samples_per_task'], baseline, states)
        result.update(status='completed' if result['judged'] == total else 'incomplete',
                      execution='parallel with target generation',
                      elapsed_seconds=time.monotonic() - started)
        short._save_summary(input_dir, result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        fcntl.flock(baseline_lock, fcntl.LOCK_UN)
        baseline_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    args = parser.parse_args()
    with (args.input_dir / '.parallel_judge.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_stream(args.input_dir)
    if result['status'] == 'incomplete':
        raise SystemExit(1)
