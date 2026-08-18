"""Smoke test: every vayas submodule stub imports cleanly."""

import importlib

import pytest

SUBMODULES = [
    "vayas",
    "vayas.protocol",
    "vayas.ingest",
    "vayas.transcribe",
    "vayas.audit",
    "vayas.metrics",
    "vayas.stats",
    "vayas.taxonomy",
    "vayas.release",
]


def test_indicvoices_r_bracket_mapping_matches_confirmed_class_labels() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from fetch_indicvoices_r_hindi import AGE_GROUP_LABELS, ELDERLY_LABEL

    assert AGE_GROUP_LABELS == ["18-30", "30-45", "45-60", "60+"]
    assert ELDERLY_LABEL in AGE_GROUP_LABELS


@pytest.mark.parametrize("module_name", SUBMODULES)
def test_submodule_imports(module_name: str) -> None:
    importlib.import_module(module_name)
