"""Validate completed batches and combine their full records for dataset upload."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dependency_alignment.document_preview import load_corpus, load_tracker, output_key, tracker_lock
from dependency_alignment.pilot import PROJECT_ROOT, digest


def combine(inputs: list[Path], output: Path, expected_count: int) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cases_raw = (PROJECT_ROOT / "data/cases.jsonl").read_bytes()
    cases = [json.loads(line) for line in cases_raw.decode().splitlines()]
    by_id = {case["case_id"]: case for case in cases}
    template = (PROJECT_ROOT / "prompts/document_dependence.txt").read_text()
    output = output.resolve()
    parquet = output.with_suffix(".parquet")
    if output.exists() or parquet.exists():
        raise ValueError("Combined output already exists; choose a fresh output path")
    if output in {path.resolve() for path in inputs}:
        raise ValueError("Combined output must differ from the source batches")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    temporary_parquet = parquet.with_suffix(".parquet.tmp")
    if temporary.exists() or temporary_parquet.exists():
        raise ValueError("A temporary export already exists; choose a fresh output path")
    sources, seen = [], set()
    try:
        with tracker_lock(PROJECT_ROOT / "configs/document_sampling.json"):
            tracker = load_tracker(PROJECT_ROOT / "configs/document_sampling.json", cases)
            with temporary.open("xb") as target:
                for path in inputs:
                    batch = tracker["corpus_batches"][output_key(path)]
                    if batch["status"] != "completed":
                        raise ValueError(f"Batch is not complete: {path}")
                    if (batch["spec"]["case_pool_sha256"] != digest(cases_raw)
                            or batch["spec"]["template_sha256"] != digest(template.encode())):
                        raise ValueError(f"Batch sources differ from the saved case pool or template: {path}")
                    completed, raw, tail = load_corpus(path, batch, by_id, template)
                    if tail or len(completed) != batch["spec"]["count"]:
                        raise ValueError(f"Incomplete source batch: {path}")
                    pairs = set(completed.values())
                    if seen.intersection(pairs):
                        raise ValueError("Source batches repeat a case–genre combination")
                    seen.update(pairs)
                    target.write(raw)
                    sources.append({"file": path.name, "documents": len(completed),
                                    "sha256": digest(raw), "sampling_seed": batch["seed"]})
            if len(seen) != expected_count:
                raise ValueError(f"Expected {expected_count} documents, found {len(seen)}")

        writer = None
        try:
            with temporary.open() as handle:
                chunk = []
                for line in handle:
                    row = json.loads(line)
                    # Variable provider metadata remains losslessly recoverable
                    # without creating thousands of dynamic Arrow struct fields.
                    for field in ("case_provenance", "provenance"):
                        row[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True)
                    chunk.append(row)
                    if len(chunk) == 256:
                        table = pa.Table.from_pylist(chunk, schema=writer.schema if writer else None)
                        if writer is None:
                            writer = pq.ParquetWriter(temporary_parquet, table.schema, compression="zstd")
                        writer.write_table(table)
                        chunk = []
                if chunk:
                    table = pa.Table.from_pylist(chunk, schema=writer.schema if writer else None)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary_parquet, table.schema, compression="zstd")
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        checked = 0
        with temporary.open() as original:
            for arrow_batch in pq.ParquetFile(temporary_parquet).iter_batches(batch_size=256):
                for row in arrow_batch.to_pylist():
                    for field in ("case_provenance", "provenance"):
                        row[field] = json.loads(row[field])
                    if row != json.loads(next(original)):
                        raise ValueError("Parquet round-trip changed a document record")
                    checked += 1
            if next(original, None) is not None or checked != expected_count:
                raise ValueError("Combined exports have different row counts")
        temporary.replace(output)
        temporary_parquet.replace(parquet)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_parquet.unlink(missing_ok=True)
    return {"documents": len(seen), "sources": sources, "jsonl": str(output),
            "jsonl_sha256": digest(output.read_bytes()), "parquet": str(parquet),
            "parquet_sha256": digest(parquet.read_bytes()),
            "parquet_bytes": parquet.stat().st_size, "jsonl_bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/dependency_documents.jsonl")
    parser.add_argument("--expected-count", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(combine(args.inputs, args.output, args.expected_count)))
