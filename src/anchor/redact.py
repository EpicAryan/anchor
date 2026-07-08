from __future__ import annotations

import re

# Order matters: multi-line/specific patterns first, generic assignments last.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private-key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("secret-assignment", re.compile(
        r"""(?i)\b[\w-]*(api[_-]?key|secret|token|password|passwd|credential)[\w-]*"""
        r"""\s*[:=]\s*['"]?[^\s'"]{8,}['"]?""")),
]


def redact(text: str) -> tuple[str, int]:
    """Replace credential-shaped substrings with [REDACTED:<kind>] markers.

    Best-effort, not a guarantee — the real protection is that cloud egress
    is opt-in. This narrows the blast radius when the user does opt in.
    """
    total = 0
    for name, pattern in _PATTERNS:
        text, n = pattern.subn(f"[REDACTED:{name}]", text)
        total += n
    return text, total
