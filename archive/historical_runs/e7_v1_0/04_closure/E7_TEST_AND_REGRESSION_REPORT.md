# E7 tests and regression

## Focused E7 suite

Command: `./Production_Project/MultiModalRAG/venv/bin/python -m pytest -q tests/test_e7_integrity_robustness.py`

Return code: `0`

```text
.....                                                                    [100%]

```

## Full suite

Command: `python -m pytest -q`

Collected tests: **295**

Return code: `0`

```text
........................................................................ [ 24%]
........................................................................ [ 48%]
........................................................................ [ 73%]
........................................................................ [ 97%]
.......                                                                  [100%]
=============================== warnings summary ===============================
tests/test_startlux_runtime.py::test_startlux_fast_kernel_detector_uses_actual_bindings
  ./Production_Project/MultiModalRAG/venv/lib/python3.12/site-packages/fla/utils/_device.py:100: UserWarning: Triton is not supported on current platform, roll back to CPU.
    _cpu_device_warning()

tests/test_startlux_runtime.py: 14 warnings
  ./Production_Project/MultiModalRAG/venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

```

Jev calls: 0. Paid API calls: 0. Reserved access: false.
