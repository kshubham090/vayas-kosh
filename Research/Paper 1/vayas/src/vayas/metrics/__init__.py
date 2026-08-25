"""ASR evaluation metrics (Phase 5): WER, CER, WER-CER divergence, task
success rate, entity accuracy, substitution/insertion/deletion ratios,
speaking-rate x WER, per-speaker distributions. See docs/protocol.md and
brief.pdf sec.3 for the full metric table and stratification axes.

Implemented: WER, CER, divergence, substitution/insertion/deletion
counts, speaking-rate (see wer_cer.py). NOT implemented -- deferred, not
silently dropped: task success rate (no task-completion definition
exists for this spontaneous-speech corpus; the brief's task-based
framing was superseded by the public-data pivot, see protocol.md) and
entity accuracy (needs a Hindi NER pipeline, a scope decision not yet
made).
"""

from vayas.metrics.wer_cer import ClipMetrics, compute_clip_metrics

__all__ = ["ClipMetrics", "compute_clip_metrics"]
