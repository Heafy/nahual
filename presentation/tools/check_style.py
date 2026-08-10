#!/usr/bin/env python3
"""Objective style checks for the Nahual presentation deck.

Flags the mechanical, scriptable design-system violations that are easy to
introduce by accident (non-palette colors, italics, em dashes in copy, external
URLs). It deliberately does NOT judge layout or hierarchy, no script can; that
stays on the checklist in presentation/DESIGN.md.

Scope: presentation/index.html and presentation/css/nahual-theme.css.

Usage:
  python3 check_style.py            # check the deck, print findings
  python3 check_style.py --hook     # PostToolUse hook mode: read the tool call
                                     # JSON on stdin, only run when a file under
                                     # presentation/ was edited

Comments are blanked before scanning (so a color or dash mentioned in a comment
is not flagged) while preserving line numbers.

Exit code 2 with findings on stderr when violations are found (so the agent is
notified); 0 when clean.
"""

import json
import re
import sys
from pathlib import Path

PRESENTATION_DIR = Path(__file__).resolve().parent.parent
HTML = PRESENTATION_DIR / "index.html"
CSS = PRESENTATION_DIR / "css" / "nahual-theme.css"

# Palette (lowercase, no leading #). The only hex colors allowed in HTML/CSS.
PALETTE = {
    "fd5108",
    "fe7c39",
    "ffaa72",  # core + sequential viz ramp
    "ffffff",
    "ffcda8",
    "ffe8d4",
    "fff5ed",  # backgrounds
    "000000",
    "595959",  # text
    "a1a8b3",  # line grey (non-text)
    "ff9f00",  # amber (tertiary)
    "059669",
    "e9b01f",
    "dc2626",  # status
    "000",
    "fff",  # tolerated neutral shorthands
}

HEX_RE = re.compile(r"#([0-9a-fA-F]{3,8})\b")
EM_DASH = "—"


def blank_comments(text, is_html):
    """Replace comment bodies with spaces (keeping newlines) so scans ignore
    comments while line numbers stay accurate."""
    pattern = r"<!--.*?-->" if is_html else r"/\*.*?\*/"

    def repl(match):
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return re.sub(pattern, repl, text, flags=re.DOTALL)


def check_palette(name, text):
    findings = []
    for match in HEX_RE.finditer(text):
        value = match.group(1).lower()
        if value not in PALETTE:
            line = text[: match.start()].count("\n") + 1
            findings.append(f"{name}:{line}: non-palette color #{value}")
    return findings


def check_italics(name, text):
    findings = []
    for index, line in enumerate(text.splitlines(), 1):
        if re.search(r"font-style\s*:\s*italic", line):
            findings.append(f"{name}:{index}: font-style: italic")
        if re.search(r"<(em|i)\b", line):
            findings.append(f"{name}:{index}: <em>/<i> tag (renders italic)")
    return findings


def check_external(name, text):
    findings = []
    for index, line in enumerate(text.splitlines(), 1):
        if re.search(r"https?://|//cdn|@import\s+url", line):
            findings.append(f"{name}:{index}: external URL / CDN reference")
    return findings


def check_em_dash(name, text):
    findings = []
    for index, line in enumerate(text.splitlines(), 1):
        if EM_DASH in line:
            findings.append(f"{name}:{index}: em dash in visible copy")
    return findings


def run_checks():
    findings = []
    for path in (HTML, CSS):
        if not path.exists():
            continue
        is_html = path == HTML
        text = blank_comments(path.read_text(encoding="utf-8"), is_html)
        name = str(path.relative_to(PRESENTATION_DIR.parent))
        findings += check_palette(name, text)
        findings += check_italics(name, text)
        findings += check_external(name, text)
        if is_html:
            findings += check_em_dash(name, text)
    return findings


def edited_path_from_stdin():
    """Return the file_path from a PostToolUse hook payload, or None."""
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return None
    return payload.get("tool_input", {}).get("file_path", "")


def main():
    if "--hook" in sys.argv:
        edited = edited_path_from_stdin()
        # Only run when a presentation file was edited.
        if not edited or "/presentation/" not in edited.replace("\\", "/"):
            return 0

    findings = run_checks()
    if findings:
        sys.stderr.write("Nahual deck style check found issues:\n")
        for finding in findings:
            sys.stderr.write(f"  - {finding}\n")
        sys.stderr.write("See presentation/DESIGN.md for the rules.\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
