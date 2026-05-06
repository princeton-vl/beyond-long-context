from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

from calibrated_memory.backend.decoder.decoder import MemoryBankDecoder
from calibrated_memory.data.sequences.collator import build_collate
from calibrated_memory.data.sequences.common import IGNORE_INDEX, TOKEN_OFFSET, YES_TOKEN, NO_TOKEN
from calibrated_memory.evaluation.checkpoint import instantiate_model_from_run, load_run_metadata


HELP_TEXT = """Commands:
  next / n           -> sample a new stream (dataset if available, otherwise random)
  len <N>            -> generate a random stream of length N
  stream <a,b,c>     -> set the stream explicitly (comma-separated integers)
  show               -> print the current stream
  help / h           -> show this message
  quit / q           -> exit the program

Any other input is interpreted as a query (comma-separated integers) whose result will be
printed as YES/NO probabilities for membership tasks or predicted continuations for
continuation tasks.
"""
def generate_random_stream(length: int, vocab_upper: int) -> List[int]:
    if length <= 0:
        raise ValueError("Stream length must be positive")
    high = max(vocab_upper, TOKEN_OFFSET + 5)
    tokens = torch.randint(TOKEN_OFFSET, high, (length,), dtype=torch.long)
    return tokens.tolist()


def tokens_to_string(tokens: List[int]) -> str:
    return " ".join(str(tok - TOKEN_OFFSET) for tok in tokens)


def parse_token_list(raw: str) -> List[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("No tokens provided")
    try:
        values = [int(part) for part in parts]
    except ValueError as err:
        raise ValueError("Tokens must be integers") from err
    return [value + TOKEN_OFFSET for value in values]


def contains_subsequence(stream_tokens: List[int], query_tokens: List[int]) -> bool:
    if not query_tokens or len(query_tokens) > len(stream_tokens):
        return False
    limit = len(stream_tokens) - len(query_tokens) + 1
    for start in range(limit):
        if stream_tokens[start : start + len(query_tokens)] == query_tokens:
            return True
    return False


def build_sample(stream_tokens: List[int], query_tokens: List[int], task: str, cont_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = list(stream_tokens)
    labels = [IGNORE_INDEX] * len(sequence)
    stream_len = len(stream_tokens)
    if task == "continuation":
        block = [0] + query_tokens + [0] + [0] * cont_len
    else:
        block = [0] + query_tokens + [0]
    sequence.extend(block)
    labels.extend([IGNORE_INDEX] * len(block))
    return (
        torch.tensor(sequence, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(stream_len, dtype=torch.long),
    )


def move_batch_to_device(batch: Dict[str, Dict[str, torch.Tensor]], device: torch.device) -> None:
    sequence = batch.get("sequence")
    if sequence:
        for key, tensor in sequence.items():
            if isinstance(tensor, torch.Tensor):
                sequence[key] = tensor.to(device)
    labels = batch.get("labels")
    if isinstance(labels, torch.Tensor):
        batch["labels"] = labels.to(device)


def run_membership_query(
    model: MemoryBankDecoder,
    collate_fn,
    pad_id: int,
    stream_tokens: List[int],
    query_tokens: List[int],
    device: torch.device,
) -> tuple[float, float]:
    sample = build_sample(stream_tokens, query_tokens, task="membership", cont_len=0)
    batch = collate_fn([sample])
    move_batch_to_device(batch, device)
    metadata = batch.get("metadata") or [{}]
    logits, class_labels = model.compute_sequence_logits(
        batch["sequence"],
        batch["labels"],
        metadata,
    )
    mask = class_labels != IGNORE_INDEX
    targets = torch.nonzero(mask[0], as_tuple=False).view(-1)
    if targets.numel() == 0:
        raise RuntimeError("Interactive query did not produce any labeled positions.")
    target_index = int(targets[-1].item())
    scores = logits[0, target_index]
    probs = torch.softmax(scores, dim=-1)
    yes_idx = model._label_to_index[YES_TOKEN]
    no_idx = model._label_to_index[NO_TOKEN]
    return float(probs[yes_idx].item()), float(probs[no_idx].item())


def run_continuation_query(
    model: MemoryBankDecoder,
    collate_fn,
    pad_id: int,
    stream_tokens: List[int],
    query_tokens: List[int],
    cont_len: int,
    device: torch.device,
) -> List[tuple[int, float]]:
    sample = build_sample(stream_tokens, query_tokens, task="continuation", cont_len=cont_len)
    batch = collate_fn([sample])
    move_batch_to_device(batch, device)
    metadata = batch.get("metadata") or [{}]
    logits, class_labels = model.compute_sequence_logits(
        batch["sequence"],
        batch["labels"],
        metadata,
    )
    mask = class_labels != IGNORE_INDEX
    targets = torch.nonzero(mask[0], as_tuple=False).view(-1)
    if targets.numel() < cont_len:
        raise RuntimeError("Continuation query produced fewer labeled tokens than requested.")
    positions = targets[-cont_len:]
    slice_logits = logits[0, positions, :]
    probs = torch.softmax(slice_logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    return [
        (int(pred.item()), float(probs[i, pred].item()))
        for i, pred in enumerate(preds)
    ]


def sample_dataset_stream(dataset, rng: random.Random) -> List[int]:
    idx = rng.randrange(len(dataset))
    seq, _labels, stream_len_tensor = dataset[idx]
    length = int(stream_len_tensor.item())
    return seq[:length].tolist()


def interactive_loop(
    model: MemoryBankDecoder,
    dataset_artifacts,
    dataset_overrides: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    sample_length: int,
) -> None:
    dataset = dataset_artifacts.dataset
    collate_fn = build_collate(dataset_artifacts.pad_id)
    rng = random.Random()
    task = str(dataset_overrides.get("task", "membership"))
    cont_len = int(dataset_overrides.get("cont_len", 3))
    vocab_upper = dataset_artifacts.pad_id

    if dataset is not None and len(dataset) > 0:
        current_stream = sample_dataset_stream(dataset, rng)
    else:
        current_stream = generate_random_stream(sample_length, vocab_upper)

    print("Loaded checkpoint from", run_dir)
    print("Type 'help' to see available commands.\n")

    while True:
        print(f"Current stream ({len(current_stream)} tokens): {tokens_to_string(current_stream)}")
        user = input("Query> ").strip()
        if not user:
            continue
        lowered = user.lower()
        if lowered in {"quit", "q"}:
            break
        if lowered in {"help", "h"}:
            print(HELP_TEXT)
            continue
        if lowered in {"next", "n"}:
            if dataset is not None and len(dataset) > 0:
                current_stream = sample_dataset_stream(dataset, rng)
            else:
                current_stream = generate_random_stream(sample_length, vocab_upper)
            continue
        if lowered.startswith("len "):
            try:
                length = int(lowered.split()[1])
            except (IndexError, ValueError):
                print("Usage: len <positive integer>")
                continue
            if length <= 0:
                print("Length must be positive.")
                continue
            current_stream = generate_random_stream(length, vocab_upper)
            continue
        if lowered.startswith("stream "):
            payload = user.split(" ", 1)[1].strip()
            try:
                current_stream = parse_token_list(payload)
            except ValueError as err:
                print(f"Invalid stream: {err}")
            continue
        if lowered == "show":
            print(tokens_to_string(current_stream))
            continue

        try:
            query_tokens = parse_token_list(user)
        except ValueError as err:
            print(f"Invalid query: {err}")
            continue

        try:
            if task == "continuation":
                predictions = run_continuation_query(
                    model,
                    collate_fn,
                    dataset_artifacts.pad_id,
                    current_stream,
                    query_tokens,
                    cont_len,
                    device,
                )
                decoded = [f"{tok - TOKEN_OFFSET} (p={prob:.3f})" for tok, prob in predictions]
                print("Predicted continuation:", ", ".join(decoded))
            elif task == "membership":
                yes_prob, no_prob = run_membership_query(
                    model,
                    collate_fn,
                    dataset_artifacts.pad_id,
                    current_stream,
                    query_tokens,
                    device,
                )
                label = "YES" if yes_prob >= no_prob else "NO"
                truth = "YES" if contains_subsequence(current_stream, query_tokens) else "NO"
                print(
                    f"Prediction: {label} (P(yes)={yes_prob:.3f}, P(no)={no_prob:.3f}) | Ground truth: {truth}"
                )
            else:
                raise ValueError(f"Unsupported task {task!r}; interactive mode only supports membership/continuation.")
        except RuntimeError as err:
            print(f"Failed to evaluate query: {err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive QA memory demo")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing config.json and checkpoints.")
    parser.add_argument(
        "--checkpoint-name",
        default="best.ckpt",
        help="Checkpoint filename inside the run directory (e.g., best.ckpt or last.ckpt).",
    )
    parser.add_argument("--device", default="auto", help="Device spec for loading the model (cpu, cuda, etc.).")
    parser.add_argument(
        "--random-stream-length",
        type=int,
        default=64,
        help="Fallback length used when sampling random streams.",
    )
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


if __name__ == "__main__":
    cli_args = parse_args()
    run_dir = cli_args.run_dir
    metadata = load_run_metadata(run_dir)
    device = resolve_device(cli_args.device)
    model, dataset_overrides, dataset_artifacts, _ = instantiate_model_from_run(
        run_dir,
        cli_args.checkpoint_name,
        device,
        metadata=metadata,
    )
    interactive_loop(
        model,
        dataset_artifacts,
        dataset_overrides,
        run_dir,
        device,
        sample_length=cli_args.random_stream_length,
    )
