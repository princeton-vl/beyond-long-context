#!/usr/bin/env python3
"""
Run anomaly detection eval: feed video + yes/no question to a VLM, extract answer.

Usage:
  python run_anomaly_eval.py questions_dataset.json --model internvl-3-5-thinking --asset-root /path/to/videos
"""
import argparse, json, os, sys, re, csv
from pathlib import Path


def load_model(model_type, max_gpu_mem=None):
    """Load a VLM model."""
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.memory_utils import calculate_max_gpu_mem as calc_mem

    if max_gpu_mem is None:
        try:
            max_gpu_mem = calc_mem(model_type)
        except Exception:
            max_gpu_mem = 80.0

    from frame_samplers import get_frame_sampler
    frame_sampler = get_frame_sampler(model_type)

    # Build kwargs
    kwargs = {'enable_metrics': False, 'max_gpu_mem': max_gpu_mem}

    # Handle thinking variants
    if 'thinking' in model_type:
        base_type = model_type.replace('-thinking', '')
        kwargs['thinking'] = True
    else:
        base_type = model_type

    from main import load_model_class
    model_class, actual_type = load_model_class(base_type)
    model = model_class(**kwargs)

    return model, frame_sampler


def extract_yes_no(response):
    """Extract yes/no from model response."""
    resp = response.strip().lower()

    # Check for {0}/{1} format (some models use this)
    if '{0}' in resp or resp.startswith('0') or resp == '0':
        return 'yes'
    if '{1}' in resp or resp.startswith('1') or resp == '1':
        return 'no'

    # Check for explicit yes/no
    if resp.startswith('yes') or 'yes' in resp[:20]:
        return 'yes'
    if resp.startswith('no') or resp[:5].strip() == 'no':
        return 'no'

    # Check for curly brace format
    m = re.search(r'\{(\w+)\}', resp)
    if m:
        val = m.group(1).lower()
        if val in ('yes', '0', 'true'):
            return 'yes'
        if val in ('no', '1', 'false'):
            return 'no'

    return 'uncertain'


def build_prompt(question_text):
    """Build the prompt for the model."""
    return (
        f"Watch this video carefully from beginning to end. "
        f"Then answer the following question with ONLY 'yes' or 'no'.\n\n"
        f"Question: {question_text}\n\n"
        f"Answer (yes or no):"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to questions_dataset.json')
    parser.add_argument('--model', required=True, help='Model type')
    parser.add_argument('--asset-root', required=True, help='Root dir for video paths')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--max-frames', type=int, default=512)
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--output-csv', default=None, help='Output CSV path')
    parser.add_argument('--state-file', default=None, help='State file for resume')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Load dataset
    dataset = json.load(open(args.dataset))
    videos = dataset.get('videos', [])
    print(f"Loaded {len(videos)} videos, {sum(len(v['questions']) for v in videos)} questions")

    # Load state for resume
    completed = set()
    if args.resume and args.state_file and os.path.exists(args.state_file):
        state = json.load(open(args.state_file))
        completed = set(state.get('completed', []))
        print(f"Resuming: {len(completed)} already completed")

    # Load model
    print(f"Loading model: {args.model}")
    model, frame_sampler = load_model(args.model)
    print("Model loaded.")

    # Run eval
    results = []
    correct = total = 0

    for vi, video in enumerate(videos):
        video_path = os.path.join(args.asset_root, video['video_path'])
        if not os.path.exists(video_path):
            print(f"  SKIP {video['video_path']}: file not found")
            continue

        # Load video frames
        try:
            video_frames = frame_sampler.sample_frames(
                video_path, fps=args.fps, max_frames=args.max_frames)
            n_frames = frame_sampler.get_frame_count(video_frames)
        except Exception as e:
            print(f"  SKIP {video['video_path']}: {e}")
            continue

        for qi, q in enumerate(video['questions']):
            qid = f"v{video['video_index']}_q{qi}"
            if qid in completed:
                continue

            prompt = build_prompt(q['question'])
            gt = q['answer']

            try:
                response = model.ask_question(
                    prompt,
                    current_video_time=float(n_frames),
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                response = f"ERROR: {e}"

            predicted = extract_yes_no(response)
            is_correct = predicted == gt
            correct += int(is_correct)
            total += 1

            result = {
                'video_file': video['video_path'],
                'video_index': video['video_index'],
                'bucket': video.get('bucket', ''),
                'contains_anomaly': video.get('contains_anomaly', False),
                'source_dataset': video.get('source_dataset', ''),
                'question': q['question'],
                'ground_truth': gt,
                'predicted': predicted,
                'is_correct': is_correct,
                'raw_response': response[:200],
                'n_frames': n_frames,
            }
            results.append(result)
            completed.add(qid)

            icon = '✅' if is_correct else '❌'
            if args.verbose:
                print(f"  {icon} [{qid}] Q: {q['question'][:60]}  GT={gt} Pred={predicted} ({n_frames}f)")

            # Save state periodically
            if args.state_file and total % 10 == 0:
                with open(args.state_file, 'w') as f:
                    json.dump({'completed': list(completed)}, f)

        if (vi + 1) % 10 == 0:
            acc = correct / total * 100 if total > 0 else 0
            print(f"  Progress: {vi+1}/{len(videos)} videos, {total} questions, acc={acc:.1f}%")

    # Final state save
    if args.state_file:
        with open(args.state_file, 'w') as f:
            json.dump({'completed': list(completed)}, f)

    # Save CSV
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
            w.writeheader()
            w.writerows(results)

    # Print summary
    acc = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{total} correct ({acc:.1f}%)")

    # Per-bucket breakdown
    from collections import defaultdict
    bucket_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        b = r['bucket']
        bucket_stats[b]['total'] += 1
        bucket_stats[b]['correct'] += int(r['is_correct'])

    print(f"\nPer-bucket accuracy:")
    for b in sorted(bucket_stats):
        s = bucket_stats[b]
        a = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {b}: {s['correct']}/{s['total']} ({a:.1f}%)")

    # Positive vs negative
    pos_correct = sum(1 for r in results if r['is_correct'] and r['contains_anomaly'])
    pos_total = sum(1 for r in results if r['contains_anomaly'])
    neg_correct = sum(1 for r in results if r['is_correct'] and not r['contains_anomaly'])
    neg_total = sum(1 for r in results if not r['contains_anomaly'])
    print(f"\n  Positive (has anomaly): {pos_correct}/{pos_total} ({pos_correct/pos_total*100:.1f}%)" if pos_total else "")
    print(f"  Negative (no anomaly): {neg_correct}/{neg_total} ({neg_correct/neg_total*100:.1f}%)" if neg_total else "")

    # Per-dataset
    ds_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        ds = r['source_dataset']
        ds_stats[ds]['total'] += 1
        ds_stats[ds]['correct'] += int(r['is_correct'])

    print(f"\nPer-dataset accuracy:")
    for ds in sorted(ds_stats):
        s = ds_stats[ds]
        a = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"  {ds}: {s['correct']}/{s['total']} ({a:.1f}%)")


if __name__ == '__main__':
    main()
