"""``THIRD_PARTY_NOTICES.md``: every file it lists carries the MIT notice, as it says.

Regression: ``rlopt/agent/wm_bptt/config.py`` (``validate_eprobe``) and
``rlopt/agent/wm_bptt/networks.py`` (``EncDecActor``) were listed as ports while
carrying no notice.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"


def _words(text: str) -> str:
    return " ".join(text.split())


def _listed_files() -> list[str]:
    rows = (
        re.match(r"\| `([^`]+)`", line) for line in NOTICES.read_text().splitlines()
    )
    return sorted({row.group(1) for row in rows if row is not None})


def _permission_notice() -> str:
    """The license's text from its permission sentence to its end."""
    block = NOTICES.read_text().split("```")[1]
    return _words(block[block.index("Permission is hereby granted") :])


def _leading_comment(path: Path) -> str:
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line[1:])
    return _words("\n".join(lines))


def test_the_notice_says_each_ported_file_carries_it():
    assert "Each ported file carries the notice." in _words(NOTICES.read_text())
    files = _listed_files()
    assert "rlopt/agent/wm_bptt/config.py" in files
    assert "rlopt/agent/wm_bptt/networks.py" in files


def test_each_listed_file_opens_with_the_notice():
    notice = _permission_notice()
    assert notice.endswith("SOFTWARE.")
    for rel in _listed_files():
        head = _leading_comment(ROOT / rel)
        assert "NDM Gym" in head, rel
        assert "05ea9f3" in head, rel
        assert "Copyright (c) 2026 NDM Gym Contributors" in head, rel
        assert notice in head, rel
