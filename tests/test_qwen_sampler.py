"""Unit tests for the Qwen frame sampler."""

import torch

from frame_samplers.qwen_sampler import QwenFrameSampler


def test_resolve_video_path_prefers_largest_candidate(tmp_path):
    """When the canonical asset is missing, use the largest matching variant."""

    missing_path = tmp_path / "pattern_video.mp4"
    # Ensure the original file is absent
    assert not missing_path.exists()

    candidate_small = tmp_path / "pattern_video_1fps_256.mp4"
    candidate_large = tmp_path / "pattern_video_2fps_384.mp4"
    candidate_small.write_bytes(b"a")
    candidate_large.write_bytes(b"ab")

    sampler = QwenFrameSampler()
    resolved = sampler._resolve_video_path(str(missing_path))

    assert resolved == str(candidate_large)


def test_sample_frames_routes_through_decord_decoder(monkeypatch, tmp_path):
    """sample_frames should resolve the path and delegate to the decord decoder."""

    fake_video = tmp_path / "pattern_video_2fps_384.mp4"
    fake_video.write_bytes(b"")

    sampler = QwenFrameSampler()

    def fake_resolve(self, path):
        return str(fake_video)

    decoded_tensor = torch.ones((3, 3, 4, 4), dtype=torch.uint8)
    decode_calls = []

    def fake_decode(self, video_path, max_frames, min_pixels, max_pixels):
        decode_calls.append(
            {
                "video_path": video_path,
                "max_frames": max_frames,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
        return decoded_tensor

    monkeypatch.setattr(QwenFrameSampler, "_resolve_video_path", fake_resolve)
    monkeypatch.setattr(QwenFrameSampler, "_decode_with_decord_all_frames", fake_decode)

    result = sampler.sample_frames(
        "/missing/pattern_video.mp4",
        fps=2,  # ignored by the sampler, but valid input
        max_frames=4,
    )

    assert torch.equal(result, decoded_tensor)
    assert len(decode_calls) == 1
    assert decode_calls[0]["video_path"] == str(fake_video)
    assert decode_calls[0]["max_frames"] == 4
    assert decode_calls[0]["min_pixels"] is None
    assert decode_calls[0]["max_pixels"] is None
