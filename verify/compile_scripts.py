#!/usr/bin/env python3
"""Compile-check packaged scripts in memory without creating .pyc files."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
scripts = sorted((ROOT / "code/scripts").glob("*.py"))
for path in scripts:
    compile(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix(), "exec")
print(f"PYTHON_SYNTAX_PASS scripts={len(scripts)}")
