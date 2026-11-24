#!/usr/bin/env python3
"""M0 read-only safety and dependency audit; never calls Jev."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from decisionrag.runtime.approval import key_presence_only

def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.yaml")
    parser.add_argument("--no-paid-api", action="store_true", required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["jev"]["enabled"] or config["jev"]["execution_allowed"]:
        raise SystemExit("unsafe config: Jev must remain disabled during M0")
    modules = ("torch", "transformers", "datasets", "sentence_transformers", "PIL", "pyarrow", "requests", "pytest", "qdrant_client")
    report = {
        "study_id": config["study_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "paid_api_mode": "NO_PAID_API",
        "jev_calls": 0,
        "jev_network_requested": False,
        "typesafe_key_present": key_presence_only(),
        "module_availability": {name: importlib.util.find_spec(name) is not None for name in modules},
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv"]),
        "git_status": command_output(["git", "-C", str(ROOT.parent), "status", "--short", "--branch"]),
        "service_probes": "not executed by sandbox preflight; use host-enabled command if unavailable",
    }
    out = ROOT / "results/preflight/M0_PREFLIGHT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state = ROOT / "results/audit/phase_state.json"
    state.write_text(json.dumps({"current_milestone": "M0", "status": "DONE", "jev_calls": 0, "jev_execution_allowed": False, "report": str(out)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
