"""Phase 0a: source-data feasibility check (handoff sec.0a).

Downloads the Common Voice Hindi validated split from Mozilla Data
Collective (MDC) and counts distinct speakers and validated clips per age
bracket (sixties/seventies/eighties), plus a collapsed "control" bucket for
everything younger. IndicVoices-R as a fallback source is not implemented
here — add it once its age-group metadata schema is confirmed.

Prerequisites (one-time, manual, not automatable):
  1. Create a free Mozilla Data Collective account and generate an API key
     in profile settings.
  2. Open the Hindi dataset page in a browser and accept its terms —
     https://mozilladatacollective.com/datasets/cmqiod71900zgnr07uiyw57br
     The MDC API explicitly does not support terms agreement via API; the
     download call below will fail (403) until this is done once.
  3. Export the key: `export MDC_TOKEN=...` (or set it in a local,
     gitignored .env — never commit it).

Real counts from this script are the only thing that may be written into
docs/protocol.md's stratification-feasibility section or
data/metadata/speakers.csv — never invented or estimated numbers
(handoff sec.4).
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests

MDC_TOKEN_ENV_VAR = "MDC_TOKEN"
MDC_API_BASE = "https://mozilladatacollective.com/api"

# "Common Voice Scripted Speech 26.0 - Hindi" — the only Hindi dataset
# confirmed to exist on MDC as of this writing (2026-08-16). MDC's API has
# no search/list endpoint, so this ID was found via the web UI, not
# discovered programmatically; re-verify by hand if this script ever needs
# to target a newer release.
DATASET_ID = "cmqiod71900zgnr07uiyw57br"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / f"common_voice_hi_{DATASET_ID}.archive"
EXTRACT_DIR = RAW_DIR / f"common_voice_hi_{DATASET_ID}"

# Common Voice's standard decade-bucket age labels. Anything in this set is
# an elderly band; anything below it (teens..fifties) collapses to
# "control"; "nineties"/"hundreds" (rare, if present) fold into "eighties"
# per docs/protocol.md sec.3.1 ("eighties (and above, if present)"); blank
# age is reported honestly as "unspecified", never dropped silently.
ELDERLY_BRACKETS = ("sixties", "seventies", "eighties")
# Common Voice's actual metadata uses the historical misspelling
# "fourties" (not "forties") for this bracket — both accepted defensively.
YOUNGER_BRACKETS = ("teens", "twenties", "thirties", "forties", "fourties", "fifties")
OLDEST_COLLAPSE = ("nineties", "hundreds", "hundred")


@dataclass(frozen=True)
class SpeakerCount:
    bracket: str
    n_speakers: int
    n_validated_clips: int


def _require_token() -> str:
    token = os.environ.get(MDC_TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(
            f"{MDC_TOKEN_ENV_VAR} is not set. Generate an API key in your "
            "Mozilla Data Collective profile settings and export it as "
            f"{MDC_TOKEN_ENV_VAR} before running this script. See this "
            "file's module docstring for the full one-time setup."
        )
    return token


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def get_dataset_metadata(token: str) -> dict:
    resp = requests.get(f"{MDC_API_BASE}/datasets/{DATASET_ID}", headers=_auth_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_download_url(token: str) -> str:
    resp = requests.post(f"{MDC_API_BASE}/datasets/{DATASET_ID}/download", headers=_auth_headers(token), timeout=30)
    if resp.status_code == 403:
        raise RuntimeError(
            "Download forbidden (403). Most likely cause: dataset terms not "
            "yet accepted via the web UI — see this file's module docstring "
            "step 2. https://mozilladatacollective.com/datasets/" + DATASET_ID
        )
    resp.raise_for_status()
    payload = resp.json()
    url = payload.get("url") or payload.get("downloadUrl") or payload.get("presignedUrl")
    if not url:
        raise RuntimeError(f"Download session response had no recognizable URL field: {payload!r}")
    return url


def download_archive(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir, filter="data")
    else:
        raise RuntimeError(
            f"{archive_path} is neither a zip nor a tar archive — inspect it "
            "manually, the MDC archive format may have changed."
        )
    return extract_dir


def find_validated_tsv(extract_dir: Path) -> Path:
    # Exact filename only — "validated_sentences.tsv" also exists in the
    # archive but lists valid *prompt sentences*, not recorded clips, and
    # has no client_id/age columns.
    candidates = sorted(extract_dir.rglob("validated.tsv"))
    if not candidates:
        raise RuntimeError(
            f"No validated.tsv found under {extract_dir}. Common Voice's "
            "standard layout ships validated.tsv/other.tsv/invalidated.tsv "
            "at the archive root — inspect the extracted archive manually, "
            "the file naming may have changed for this release."
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple validated.tsv candidates found: {candidates} — disambiguate manually.")
    return candidates[0]


def _bracket_for(raw_age: str) -> str:
    age = raw_age.strip().lower()
    if not age:
        return "unspecified"
    if age in ELDERLY_BRACKETS:
        return age
    if age in YOUNGER_BRACKETS:
        return "control"
    if age in OLDEST_COLLAPSE:
        return "eighties"
    return f"unrecognized:{age}"


def count_speakers_by_age_bracket(validated_tsv: Path) -> list[SpeakerCount]:
    speakers_by_bracket: dict[str, set[str]] = defaultdict(set)
    clips_by_bracket: dict[str, int] = defaultdict(int)

    with open(validated_tsv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"{validated_tsv} has no header row.")
        missing = {"client_id", "age"} - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"{validated_tsv} is missing expected column(s) {missing}. "
                f"Actual columns: {reader.fieldnames}. Common Voice's schema "
                "may have changed for this release — inspect manually."
            )
        for row in reader:
            bracket = _bracket_for(row["age"])
            speakers_by_bracket[bracket].add(row["client_id"])
            clips_by_bracket[bracket] += 1

    return [
        SpeakerCount(bracket=b, n_speakers=len(speakers_by_bracket[b]), n_validated_clips=clips_by_bracket[b])
        for b in sorted(speakers_by_bracket)
    ]


def main() -> None:
    token = _require_token()

    if ARCHIVE_PATH.exists():
        print(f"Using cached archive at {ARCHIVE_PATH}", file=sys.stderr)
    else:
        meta = get_dataset_metadata(token)
        print(f"Dataset metadata: {meta}", file=sys.stderr)
        url = create_download_url(token)
        print("Downloading archive (this may take a while for 545MB)...", file=sys.stderr)
        download_archive(url, ARCHIVE_PATH)

    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)
    validated_tsv = find_validated_tsv(EXTRACT_DIR)
    counts = count_speakers_by_age_bracket(validated_tsv)

    print(f"\nSource: {validated_tsv.relative_to(RAW_DIR)}")
    print(f"{'bracket':<20}{'n_speakers':<12}{'n_validated_clips':<18}")
    for c in counts:
        print(f"{c.bracket:<20}{c.n_speakers:<12}{c.n_validated_clips:<18}")

    n_underpowered = [c for c in counts if c.bracket in ELDERLY_BRACKETS and c.n_speakers < 8]
    if n_underpowered:
        print(
            "\nSTOPPING-RULE FLAG (docs/protocol.md sec.5): bands below n=8 "
            f"speakers: {[c.bracket for c in n_underpowered]}. Report as a "
            "limitation, do not drop silently."
        )


if __name__ == "__main__":
    main()
