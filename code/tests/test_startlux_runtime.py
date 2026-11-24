from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data/models/e6_1b/startlux_decision_v1_4b"


def test_startlux_fast_kernel_detector_uses_actual_bindings():
    import sys

    sys.path.insert(0, str(CHECKPOINT))
    from startlux_decision.model import fast_kernels_active

    assert fast_kernels_active(str(CHECKPOINT)) is True


def test_startlux_checkpoint_and_native_code_are_present():
    assert (CHECKPOINT / "config.json").is_file()
    assert (CHECKPOINT / "model.safetensors.index.json").is_file()
    assert (CHECKPOINT / "startlux_decision/model.py").is_file()
