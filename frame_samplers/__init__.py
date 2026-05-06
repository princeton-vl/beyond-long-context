"""Frame sampling utilities for different video-language models."""

import os

# CRITICAL: Set backend BEFORE any qwen_vl_utils/qwen_omni_utils imports
# This prevents std::bad_alloc errors from torchcodec backend
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")


def get_frame_sampler(model_type: str):
    """Get appropriate frame sampler for model type."""
    if model_type == "minicpm":
        from .minicpm_v_2_6_sampler import MiniCPMFrameSampler
        return MiniCPMFrameSampler()
    elif model_type == "m3_agent":
        from .m3_agent_sampler import M3AgentFrameSampler
        return M3AgentFrameSampler()
    elif model_type == "timechat":
        from .timechat_sampler import TimeChatFrameSampler
        return TimeChatFrameSampler()
    elif model_type == "qwen3_omni":
        from .qwen3_omni_sampler import Qwen3oFrameSampler
        return Qwen3oFrameSampler()
    elif model_type.startswith("internvl-3-5"):
        from .internvl_sampler import InternVLFrameSampler
        return InternVLFrameSampler()
    elif model_type.startswith("minicpm-4-5"):
        from .minicpm_v_4_5_sampler import MiniCPM45FrameSampler
        return MiniCPM45FrameSampler()
    elif model_type == "phi_multimodal":
        from .phi_sampler import PhiFrameSampler
        return PhiFrameSampler()
    elif model_type == "longvila":
        from .longvila_sampler import LongVILAFrameSampler
        return LongVILAFrameSampler()
    else:
        # All other models (qwen_full, qwen_mini) use Qwen-style sampling
        from .qwen_sampler import QwenFrameSampler
        return QwenFrameSampler()
