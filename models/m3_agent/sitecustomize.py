"""M3-Agent specific sitecustomize hooks."""

def _apply_patch() -> None:
    try:
        from utils.vllm_compat import ensure_vllm_disabled_tqdm_patch
    except Exception:  # pragma: no cover - repo might not be on sys.path
        return

    try:
        ensure_vllm_disabled_tqdm_patch()
    except Exception:
        pass


_apply_patch()

