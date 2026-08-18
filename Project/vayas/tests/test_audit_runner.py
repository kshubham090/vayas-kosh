"""Phase 4 batch runner, tested against fake ASRSystem implementations —
none of the real model/API wrappers are exercised here (that needs real
weights/network/GPU, deliberately out of scope for unit tests, same as
Phase 2's audio validation was tested only against synthetic fixtures)."""

from pathlib import Path

from vayas.audit.base import ASRSystem
from vayas.audit.runner import Utterance, run_batch


class FakeASRSystem(ASRSystem):
    def __init__(self, name: str, fail_on: set[str] = frozenset()) -> None:
        self.name = name
        self._fail_on = fail_on
        self.calls: list[tuple[Path, str]] = []
        self.unload_called = False

    def transcribe(self, audio_path: Path, lang: str) -> str:
        self.calls.append((audio_path, lang))
        if audio_path.stem in self._fail_on:
            raise RuntimeError(f"simulated failure for {audio_path.stem}")
        return f"hypothesis for {audio_path.stem} [{lang}]"

    def unload(self) -> None:
        self.unload_called = True


def _utterances(tmp_path: Path, ids: list[str]) -> list[Utterance]:
    out = []
    for uid in ids:
        p = tmp_path / f"{uid}.wav"
        p.write_bytes(b"")  # never actually read by FakeASRSystem
        out.append(Utterance(utt_id=uid, audio_path=p, lang="hi"))
    return out


def test_writes_one_hypothesis_file_per_utterance_per_system(tmp_path: Path) -> None:
    system = FakeASRSystem("fake-system-a")
    utts = _utterances(tmp_path, ["utt1", "utt2"])
    output_dir = tmp_path / "hyps"

    failures = run_batch([system], utts, output_dir=output_dir)

    assert failures == []
    assert (output_dir / "fake-system-a" / "utt1.txt").read_text(encoding="utf-8") == "hypothesis for utt1 [hi]"
    assert (output_dir / "fake-system-a" / "utt2.txt").read_text(encoding="utf-8") == "hypothesis for utt2 [hi]"


def test_multiple_systems_get_separate_subdirectories(tmp_path: Path) -> None:
    system_a = FakeASRSystem("system-a")
    system_b = FakeASRSystem("system-b")
    utts = _utterances(tmp_path, ["utt1"])
    output_dir = tmp_path / "hyps"

    run_batch([system_a, system_b], utts, output_dir=output_dir)

    assert (output_dir / "system-a" / "utt1.txt").exists()
    assert (output_dir / "system-b" / "utt1.txt").exists()


def test_one_utterance_failure_does_not_abort_the_batch(tmp_path: Path) -> None:
    system = FakeASRSystem("fake-system", fail_on={"bad_utt"})
    utts = _utterances(tmp_path, ["good_utt", "bad_utt", "good_utt2"])
    output_dir = tmp_path / "hyps"

    failures = run_batch([system], utts, output_dir=output_dir)

    assert len(failures) == 1
    assert failures[0].utt_id == "bad_utt"
    assert failures[0].system_name == "fake-system"
    assert (output_dir / "fake-system" / "good_utt.txt").exists()
    assert (output_dir / "fake-system" / "good_utt2.txt").exists()
    assert not (output_dir / "fake-system" / "bad_utt.txt").exists()


def test_existing_output_is_not_recomputed(tmp_path: Path) -> None:
    system = FakeASRSystem("fake-system")
    utts = _utterances(tmp_path, ["utt1"])
    output_dir = tmp_path / "hyps"

    run_batch([system], utts, output_dir=output_dir)
    assert len(system.calls) == 1

    run_batch([system], utts, output_dir=output_dir)
    assert len(system.calls) == 1  # not called again — resumable


def test_unload_called_after_each_system_even_on_failure(tmp_path: Path) -> None:
    # Required on this project's 4GB-VRAM dev GPU: holding two systems'
    # models loaded at once risks OOM, so each system must release its
    # memory before the next one loads -- including when transcribe()
    # raised partway through.
    system_a = FakeASRSystem("system-a", fail_on={"utt1"})
    system_b = FakeASRSystem("system-b")
    utts = _utterances(tmp_path, ["utt1"])
    output_dir = tmp_path / "hyps"

    run_batch([system_a, system_b], utts, output_dir=output_dir)

    assert system_a.unload_called
    assert system_b.unload_called


def test_asr_system_is_abstract_and_requires_transcribe() -> None:
    import pytest

    with pytest.raises(TypeError):
        ASRSystem()  # type: ignore[abstract]
