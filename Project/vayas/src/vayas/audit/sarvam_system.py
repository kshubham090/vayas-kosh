"""Sarvam AI speech-to-text, zero-shot, via their REST API.

NAMING DRIFT FROM brief.pdf / docs/protocol.md: both name "Sarvam
Saarika" as the system to audit. As of this writing, Sarvam's current API
docs (docs.sarvam.ai) no longer list "saarika" as an accepted `model`
value on the /speech-to-text endpoint — only "saaras:v3" (current
default) and "saaras:v4" (latest) are documented; Saarika/Saaras v2.5 are
called out as legacy and being phased out. This wrapper defaults to
"saaras:v3" with mode="transcribe" as the closest current equivalent to
what "Saarika" meant when the brief was written. If Shubham wants the
paper to explicitly name "Saaras" instead of "Saarika" (or pin to a
specific legacy version while it's still available), that's a wording
decision for docs/protocol.md, not something this wrapper can silently
resolve.

Verified interface: POST https://api.sarvam.ai/speech-to-text,
multipart/form-data, header `api-subscription-key`, required `file`,
optional `model`/`language_code`/`mode`. Response JSON has a `transcript`
field. Requires SARVAM_API_KEY in the environment — an API-billed system,
not local weights; do not call this in bulk without confirming budget.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from vayas.audit.base import ASRSystem

API_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_API_KEY_ENV_VAR = "SARVAM_API_KEY"
_LANG_CODE_MAP = {"hi": "hi-IN"}


class SarvamSystem(ASRSystem):
    name = "sarvam-saaras-v3"  # see module docstring re: Saarika -> Saaras naming drift

    def __init__(self, model: str = "saaras:v3", mode: Optional[str] = "transcribe") -> None:
        self._model = model
        self._mode = mode
        self._api_key: Optional[str] = None

    def _require_key(self) -> str:
        if self._api_key is None:
            key = os.environ.get(SARVAM_API_KEY_ENV_VAR)
            if not key:
                raise RuntimeError(f"{SARVAM_API_KEY_ENV_VAR} is not set. This is a billed API — confirm budget first.")
            self._api_key = key
        return self._api_key

    def transcribe(self, audio_path: Path, lang: str) -> str:
        import requests

        api_key = self._require_key()
        sarvam_lang = _LANG_CODE_MAP.get(lang, lang)

        data = {"model": self._model, "language_code": sarvam_lang}
        if self._mode is not None:
            data["mode"] = self._mode

        with open(audio_path, "rb") as f:
            resp = requests.post(
                API_URL,
                headers={"api-subscription-key": api_key},
                files={"file": f},
                data=data,
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()["transcript"].strip()
