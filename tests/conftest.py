"""Pytest configuration ensuring the repository root is importable."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

if 'torch' not in sys.modules:
    fake_torch = ModuleType('torch')

    class _FakeCuda:
        OutOfMemoryError = RuntimeError

        def is_available(self) -> bool:
            return False

        def empty_cache(self) -> None:
            return None

        def reset_peak_memory_stats(self, *args, **kwargs) -> None:
            return None

    fake_torch.cuda = _FakeCuda()
    fake_torch.Tensor = SimpleNamespace  # minimal placeholder
    sys.modules['torch'] = fake_torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if 'torch._strobelight.compile_time_profiler' not in sys.modules:
    profiler_module = ModuleType('torch._strobelight.compile_time_profiler')

    class _StubProfiler:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    profiler_module.StrobelightCompileTimeProfiler = _StubProfiler
    strobelight_module = ModuleType('torch._strobelight')
    strobelight_module.compile_time_profiler = profiler_module
    sys.modules['torch._strobelight'] = strobelight_module
    sys.modules['torch._strobelight.compile_time_profiler'] = profiler_module
