"""AI4Bharat IndicWhisper (Hindi), zero-shot.

Unlike the other wrappers in this module, IndicWhisper is NOT published on
the Hugging Face Hub as a `from_pretrained("org/repo")` checkpoint — per
the official AI4Bharat/Vistaar repo (github.com/AI4Bharat/vistaar), the
Hindi model is distributed as a direct zip download from an AI4Bharat
object-store URL, in Hugging Face checkpoint format. This wrapper
downloads + extracts that zip once, then loads it locally.

CONFIRMED WORKING END-TO-END (2026-08-16), after three real issues found
and fixed by an actual run (not assumed from docs):
1. The zip's internal layout does match a standard HF checkpoint --
   find_checkpoint_dir()'s defensive config.json search was validated,
   not just written defensively.
2. `forced_decoder_ids=` is removed in this transformers version;
   replaced with `language=`/`task=` passed to generate() directly.
3. This checkpoint ships a pre-multilingual-API generation_config.json
   (real ValueError: "generation config is outdated"). Fixed by loading
   a compatible GenerationConfig from openai/whisper-large-v3 (same
   1259-tensor count -> same architecture family).

Real transcript vs. gold, same reference clip as the other verified
wrappers (common_voice_hi_26127974.mp3):
  gold: इंडिया टुडे ग्रुप ने प्रधानमंत्री को श्वेत पत्र सौंपा
  hyp:  इंडिया टूडे ग्रुप ने प्रधानमंत्री को श्वेत पत्र सौंपा
Near-exact (one vowel-length variant, टूडे/टुडे) -- markedly closer than
plain Whisper large-v3 on this clip, consistent with being a Hindi
fine-tune rather than a general multilingual model.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

from vayas.audit.base import ASRSystem, default_device, load_audio_mono_16k

HINDI_CHECKPOINT_URL = "https://indicwhisper.objectstore.e2enetworks.net/hindi_models.zip"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw" / "indicwhisper_hindi"


class IndicWhisperSystem(ASRSystem):
    name = "indicwhisper-hindi"

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device or default_device()
        self._model = None
        self._processor = None

    def _download_and_extract(self) -> Path:
        zip_path = CACHE_DIR.with_suffix(".zip")
        extract_dir = CACHE_DIR
        if not extract_dir.exists():
            import requests

            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(HINDI_CHECKPOINT_URL, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        candidates = [p.parent for p in extract_dir.rglob("config.json")]
        if not candidates:
            raise RuntimeError(
                f"No config.json found under {extract_dir} after extracting "
                f"{HINDI_CHECKPOINT_URL} — the zip's internal layout may not "
                "match a standard Hugging Face checkpoint. Inspect manually."
            )
        return candidates[0]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

        checkpoint_dir = self._download_and_extract()
        self._processor = WhisperProcessor.from_pretrained(checkpoint_dir)
        self._model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir).to(self._device)

        # This checkpoint ships a pre-multilingual-API generation_config.json
        # (confirmed by a real ValueError: "generation config is outdated
        # and is thus not compatible with the `language` argument"). Same
        # tensor count as openai/whisper-large-v3 (1259) -> same
        # architecture family, so its generation_config's lang_to_id/
        # task_to_id mappings are compatible. See
        # https://github.com/huggingface/transformers/issues/25084
        self._model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")

    def transcribe(self, audio_path: Path, lang: str) -> str:
        import torch

        self._ensure_loaded()

        wav = load_audio_mono_16k(audio_path)

        inputs = self._processor(wav.squeeze(0).numpy(), sampling_rate=16_000, return_tensors="pt")
        # forced_decoder_ids= is removed in this transformers version
        # (confirmed by a real ValueError) -- language/task are now passed
        # to generate() directly instead of pre-computing prompt ids.
        with torch.no_grad():
            predicted_ids = self._model.generate(
                inputs.input_features.to(self._device), language=lang, task="transcribe"
            )
        return self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model, self._processor
        self._model = None
        self._processor = None
        if self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
