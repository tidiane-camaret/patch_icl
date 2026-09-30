#!/usr/bin/env python3
"""Flatten a multi-file LaTeX project into one file by recursively expanding
\\input{...} and \\include{...}. Minimal latexpand replacement (no network dep).

Usage:
    python flatten_tex.py main.tex > thesis_full.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

INPUT_RE = re.compile(r"^(?P<indent>[^%]*)\\(?:input|include)\{(?P<path>[^}]+)\}")


def resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.suffix != ".tex":
        p = p.with_suffix(".tex")
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def rel(target: Path, base: Path) -> Path:
    try:
        return target.relative_to(base)
    except ValueError:
        return target


def flatten(path: Path, base: Path, seen: set[Path]) -> list[str]:
    if path in seen:
        return [f"% [flatten_tex.py] SKIPPED repeated include: {path}\n"]
    seen.add(path)
    out: list[str] = []
    text = path.read_text()
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            out.append(line)
            continue
        m = INPUT_RE.match(line)
        if m:
            target = resolve(m.group("path"), base)
            out.append(f"% >>> begin {rel(target, base)}\n")
            if target.exists():
                out.extend(flatten(target, base, seen))
            else:
                out.append(f"% [flatten_tex.py] MISSING FILE: {target}\n")
            out.append(f"% <<< end {rel(target, base)}\n")
        else:
            out.append(line)
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: flatten_tex.py <main.tex>", file=sys.stderr)
        sys.exit(1)
    entry = Path(sys.argv[1]).resolve()
    base = entry.parent
    sys.stdout.writelines(flatten(entry, base, set()))


if __name__ == "__main__":
    main()
