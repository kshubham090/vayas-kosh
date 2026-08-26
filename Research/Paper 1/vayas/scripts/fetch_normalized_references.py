"""Fetch the `normalized` transcript field for every clip already used in
this project's manifests, to test whether the paper's age-effect claims
survive scoring against a disfluency-free reference instead of
`verbatim` (Limitations: the verbatim/normalized confound flagged after
external review).

Metadata-only: reads `audio.path` (via nested column projection, which
excludes the `bytes` subfield) and `normalized` across every row group
of every shard -- no audio bytes are ever fetched. utt_id is derived
from audio.path exactly as in fetch_pilot_scale_audio.py, so it joins
directly against the existing manifest_*.csv files without needing the
original (shard, group, pos) locators.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import fsspec

BASE = "https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi/resolve/refs%2Fconvert%2Fparquet/default/train/{:04d}.parquet"
N_SHARDS = 10
MAX_RETRIES = 5
ROOT = Path(__file__).resolve().parent.parent
HYPOTHESES_DIR = ROOT / "data" / "transcripts" / "hypotheses"
OUT_PATH = HYPOTHESES_DIR / "normalized_references.csv"


def read_row_group_with_retry(shard_idx: int, row_group_idx: int, columns: list[str]):
    url = BASE.format(shard_idx)
    fs = fsspec.filesystem("https")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with fs.open(url, "rb") as f:
                pf = pq.ParquetFile(f)
                return pf.read_row_group(row_group_idx, columns=columns)
        except Exception as e:  # noqa: BLE001
            print(f"  shard {shard_idx} group {row_group_idx} attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5 * attempt)


def num_row_groups(shard_idx: int) -> int:
    url = BASE.format(shard_idx)
    fs = fsspec.filesystem("https")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with fs.open(url, "rb") as f:
                return pq.ParquetFile(f).num_row_groups
        except Exception as e:  # noqa: BLE001
            print(f"  shard {shard_idx} num_row_groups attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5 * attempt)


def load_wanted_utt_ids() -> set[str]:
    wanted = set()
    for path in sorted(HYPOTHESES_DIR.glob("manifest_*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                wanted.add(row["utt_id"])
    print(f"{len(wanted)} unique utt_ids to look up", flush=True)
    return wanted


def main() -> None:
    wanted = load_wanted_utt_ids()
    found: dict[str, str] = {}

    for shard_idx in range(N_SHARDS):
        n_groups = num_row_groups(shard_idx)
        print(f"[shard {shard_idx}] {n_groups} row groups", flush=True)
        for group_idx in range(n_groups):
            table = read_row_group_with_retry(shard_idx, group_idx, ["audio.path", "normalized"])
            paths = pc.struct_field(table.column("audio"), "path").to_pylist()
            normalized = table.column("normalized").to_pylist()
            for path, norm in zip(paths, normalized):
                utt_id = path.replace(".wav", "")
                if utt_id in wanted:
                    found[utt_id] = norm
        print(f"  running total found: {len(found)}/{len(wanted)}", flush=True)
        if len(found) == len(wanted):
            print("all utt_ids found, stopping early", flush=True)
            break

    missing = wanted - found.keys()
    if missing:
        print(f"WARNING: {len(missing)} utt_ids not found: {list(missing)[:10]}", flush=True)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "normalized"])
        for utt_id, norm in found.items():
            w.writerow([utt_id, norm])

    print(f"Wrote {len(found)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
