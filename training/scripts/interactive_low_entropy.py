"""Interactive membership probe for low-entropy checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.common import IGNORE_INDEX, TOKEN_OFFSET, YES_TOKEN, NO_TOKEN, UNCERTAIN_TOKEN
from calibrated_memory.data.sequences.question_generator import build_samples
from calibrated_memory.evaluation.checkpoint import instantiate_model_from_run, load_run_metadata


def _default_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise SystemExit("A CUDA device is required for interactive probing (none detected).")


def _sample_stream(generator: torch.Generator, length: int, token_offset: int, vocab_high: int) -> list[int]:
    if length <= 0:
        raise ValueError("Sequence length must be positive.")
    return torch.randint(token_offset, vocab_high, (length,), generator=generator).tolist()


def _contains_subsequence(stream: Sequence[int], candidate: Sequence[int]) -> bool:
    if not candidate or len(candidate) > len(stream):
        return False
    limit = len(stream) - len(candidate) + 1
    for start in range(limit):
        if stream[start : start + len(candidate)] == list(candidate):
            return True
    return False


def _build_membership_sample(stream: Sequence[int], candidate: Sequence[int], truth: bool) -> tuple:
    def query_fn(_: Sequence[int]) -> list[tuple[list[int], bool]]:
        return [(list(candidate), truth)]

    samples, _, _, _ = build_samples([list(stream)], query_fn, task="membership")
    if not samples:
        raise RuntimeError("Failed to build a membership sample for the provided stream.")
    return samples[0]


def _move_sequence_batch_to_device(batch: dict, device: torch.device) -> None:
    sequence = batch.get("sequence")
    if isinstance(sequence, dict):
        for key, tensor in sequence.items():
            if isinstance(tensor, torch.Tensor):
                sequence[key] = tensor.to(device)
    if isinstance(batch.get("labels"), torch.Tensor):
        batch["labels"] = batch["labels"].to(device)


def _token_list_display(tokens: Sequence[int]) -> str:
    return " ".join(str(token) for token in tokens)


def _describe_prediction(token: int) -> str:
    if token == YES_TOKEN:
        return "yes"
    if token == NO_TOKEN:
        return "no"
    if token == UNCERTAIN_TOKEN:
        return "uncertain"
    return f"token({token})"


def _parse_candidate(raw: Sequence[str], vocab_limit: int) -> list[int]:
    if not raw:
        raise ValueError("Provide at least one number for the candidate sequence.")
    try:
        values = [int(token) for token in raw]
    except ValueError as exc:  # noqa: BLE001
        raise ValueError("Candidate tokens must be integers.") from exc
    for value in values:
        if value < 0 or value >= vocab_limit:
            raise ValueError(
                f"Candidate tokens must lie in [0, {vocab_limit - 1}]; got {value}."
            )
    return values


def interactive_probe(args: argparse.Namespace) -> None:
    metadata = load_run_metadata(args.run_dir)
    device = _default_device(args.device)
    model, dataset_overrides, dataset_artifacts, _ = instantiate_model_from_run(
        args.run_dir,
        args.checkpoint_name,
        device,
        metadata=metadata,
    )
    token_offset = int(dataset_overrides.get("token_offset", TOKEN_OFFSET))
    vocab_high = int(dataset_artifacts.pad_id)
    vocab_span = max(1, vocab_high - token_offset)
    collate_fn = build_collate(dataset_artifacts.pad_id)
    rng = torch.Generator().manual_seed(args.seed)

    def _new_stream() -> list[int]:
        return _sample_stream(rng, args.seq_len, token_offset, vocab_high)

    current_stream = _new_stream()
    print("Interactive low-entropy membership probe ready.")
    print("Commands: 'n' new stream, 's' show stream, 'q <space-separated ints>' query, 'help', 'exit'.")
    while True:
        try:
            raw = input(": ").strip()
        except EOFError:
            print()
            break
        if not raw:
            continue
        parts = raw.split()
        command = parts[0].lower()
        if command in {"exit", "quit"}:
            break
        if command in {"help", "?"}:
            print("n: new stream, s: show stream, q <ints>: query membership, exit: quit.")
            continue
        if command in {"n", "new"}:
            current_stream = _new_stream()
            print("Sampled a new stream.")
            continue
        if command in {"s", "show"}:
            decoded = [token - token_offset for token in current_stream]
            print("Stream tokens:")
            print(_token_list_display(decoded))
            continue
        if command in {"q", "query"}:
            candidate_entries = parts[1:]
            try:
                candidate_decoded = _parse_candidate(candidate_entries, vocab_span)
            except ValueError as exc:
                print(f"[error] {exc}")
                continue
            candidate_tokens = [value + token_offset for value in candidate_decoded]
            stream_decoded = [token - token_offset for token in current_stream]
            truth = _contains_subsequence(stream_decoded, candidate_decoded)
            sample = _build_membership_sample(current_stream, candidate_tokens, truth)
            batch = collate_fn([sample])
            _move_sequence_batch_to_device(batch, device)
            model.eval()
            metadata_entries = batch.get("metadata") or [{}]
            with torch.no_grad():
                logits, labels = model.compute_sequence_logits(
                    batch["sequence"],
                    batch["labels"],
                    metadata_entries,
                )
            label_mask = labels[0] != IGNORE_INDEX
            valid_positions = torch.nonzero(label_mask, as_tuple=False).view(-1)
            if valid_positions.numel() == 0:
                print("[warn] Query encoding produced no label positions; skipping.")
                continue
            query_logits = logits[0][valid_positions][0]
            predicted_index = int(torch.argmax(query_logits).item())
            predicted_token = model._class_index_to_token(predicted_index)
            probs = torch.softmax(query_logits, dim=-1)
            pred_prob = float(probs[predicted_index].item())
            print(
                f"Model prediction: {_describe_prediction(predicted_token)} (p={pred_prob:.3f}) | "
                f"truth={'yes' if truth else 'no'}"
            )
            continue
        print("Unrecognized command; type 'help' for options.")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive low-entropy membership probe.")
    parser.add_argument("--run-dir", type=str, required=True, help="Checkpoint run directory (config.json parent).")
    parser.add_argument(
        "--checkpoint-name",
        default="best-overall.ckpt",
        help="Checkpoint filename to load from --run-dir.",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Length of the synthetic streams to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic stream generation.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device spec (auto selects CUDA).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    interactive_probe(args)


if __name__ == "__main__":
    main()
