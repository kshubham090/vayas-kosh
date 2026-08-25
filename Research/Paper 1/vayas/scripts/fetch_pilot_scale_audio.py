"""Download real audio for the scaled batch run: all 50 elderly speakers'
utterances + a duration-matched control sample, from IndicVoices-R Hindi.

Two-pass, memory-safe design (fixes a real bug from the first attempt at
this: holding every control-pool clip's audio bytes in RAM at once --
roughly 22,000 clips x ~1.5MB, tens of GB -- caused repeated `realloc`
failures partway through). Pass 1 reads only cheap metadata columns
(no audio) across all shards to decide exactly which rows are needed.
Pass 2 re-reads only the row groups that actually contain a wanted row,
extracts just those rows' audio, and writes straight to disk -- at most
one row group's worth of audio (~100-200MB) is ever held in memory.

Writes WAV files to data/raw/batch_audio/ plus a manifest.csv (utt_id,
speaker_id, age_band, duration, verbatim). Retries transient network/SSL
errors per shard (observed repeatedly on large fetches from this host).
"""

from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import fsspec

AGE_LABELS = ["18-30", "30-45", "45-60", "60+"]
BASE = "https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi/resolve/refs%2Fconvert%2Fparquet/default/train/{:04d}.parquet"
N_SHARDS = 10
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "batch_audio"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
MAX_RETRIES = 5

random.seed(42)  # reproducible control sample


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Pass 1: metadata only, no audio ----------
    elderly_locators = []  # (shard, group, pos, speaker_id, duration)
    control_locators = []

    for shard_idx in range(N_SHARDS):
        print(f"[pass 1] scanning shard {shard_idx} metadata...", flush=True)
        n_groups = num_row_groups(shard_idx)
        for group_idx in range(n_groups):
            table = read_row_group_with_retry(shard_idx, group_idx, ["speaker_id", "age_group", "duration"])
            sids = table.column("speaker_id").to_pylist()
            ages = table.column("age_group").to_pylist()
            durations = table.column("duration").to_pylist()
            for pos, (sid, a, dur) in enumerate(zip(sids, ages, durations)):
                label = AGE_LABELS[a]
                locator = (shard_idx, group_idx, pos, sid, dur)
                if label == "60+":
                    elderly_locators.append(locator)
                else:
                    control_locators.append(locator)
        print(f"  running totals: elderly={len(elderly_locators)} control_pool={len(control_locators)}", flush=True)

    elderly_duration = sum(loc[4] for loc in elderly_locators)
    print(f"\n[pass 1 done] elderly: {len(elderly_locators)} clips, {elderly_duration/3600:.2f}h total", flush=True)

    # Duration-matched control sample -- cheap, metadata only.
    random.shuffle(control_locators)
    control_sample = []
    running = 0.0
    for loc in control_locators:
        if running >= elderly_duration:
            break
        control_sample.append(loc)
        running += loc[4]
    print(f"Control sample: {len(control_sample)} clips, {running/3600:.2f}h total", flush=True)

    wanted = elderly_locators + control_sample
    age_band_by_key = {(s, g, p): "60+" for s, g, p, *_ in elderly_locators}
    age_band_by_key.update({(s, g, p): "control" for s, g, p, *_ in control_sample})

    # Group wanted positions by (shard, row_group) so pass 2 only touches
    # the row groups that actually contain a wanted row.
    wanted_by_group: dict[tuple[int, int], set[int]] = defaultdict(set)
    for shard_idx, group_idx, pos, _sid, _dur in wanted:
        wanted_by_group[(shard_idx, group_idx)].add(pos)

    print(f"\n[pass 2] fetching audio for {len(wanted)} clips across {len(wanted_by_group)} row groups...", flush=True)

    manifest = []
    for (shard_idx, group_idx), positions in sorted(wanted_by_group.items()):
        table = read_row_group_with_retry(
            shard_idx, group_idx, ["speaker_id", "age_group", "audio", "verbatim", "duration"]
        )
        sids = table.column("speaker_id").to_pylist()
        ages = table.column("age_group").to_pylist()
        audios = table.column("audio").to_pylist()
        verbatims = table.column("verbatim").to_pylist()
        durations = table.column("duration").to_pylist()

        for pos in positions:
            sid, dur, verbatim, audio = sids[pos], durations[pos], verbatims[pos], audios[pos]
            age_band = age_band_by_key[(shard_idx, group_idx, pos)]
            utt_id = audio["path"].replace(".wav", "")
            out_path = OUT_DIR / f"{utt_id}.wav"
            if not out_path.exists():
                out_path.write_bytes(audio["bytes"])
            manifest.append({
                "utt_id": utt_id, "speaker_id": sid, "age_band": age_band,
                "duration": dur, "verbatim": verbatim,
            })
        # `table` (and its audio bytes) goes out of scope at the next loop
        # iteration -- at most one row group's audio is ever held at once.
        print(f"  shard {shard_idx} group {group_idx}: wrote {len(positions)} clips "
              f"({len(manifest)}/{len(wanted)} total)", flush=True)

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "speaker_id", "age_band", "duration", "verbatim"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nWrote {len(manifest)} clips to {OUT_DIR}", flush=True)
    print(f"Manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
