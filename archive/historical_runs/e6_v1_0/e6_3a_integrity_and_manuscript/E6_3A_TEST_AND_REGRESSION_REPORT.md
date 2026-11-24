# E6-3A Test and Regression Report

```json
{
  "focused": {
    "command": "./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q tests/test_e6_3a_integrity_and_manuscript.py",
    "returncode": 0,
    "passed": 8,
    "output": "........                                                                 [100%]\n"
  },
  "full_regression": {
    "command": "./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q",
    "returncode": 0,
    "passed": 282,
    "collected": 282,
    "output": "........................................................................ [ 25%]\n........................................................................ [ 51%]\n........................................................................ [ 76%]\n..................................................................       [100%]\n=============================== warnings summary ===============================\ntests/test_startlux_runtime.py::test_startlux_fast_kernel_detector_uses_actual_bindings\n  ./Production_Project/MultiModalRAG/venv/lib/python3.12/site-packages/fla/utils/_device.py:100: UserWarning: Triton is not supported on current platform, roll back to CPU.\n    _cpu_device_warning()\n\ntests/test_startlux_runtime.py: 14 warnings\n  ./Production_Project/MultiModalRAG/venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.\n    warnings.warn(\n\n-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
  },
  "historical_baseline": "E6-3 recorded 274/274 in MultiModalRAG/venv; this run records the current authoritative suite result",
  "jev_calls": 0,
  "paid_api_calls": 0,
  "reserved_access": false
}
```
