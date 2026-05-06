from __future__ import annotations
import argparse
import json
from typing import Any, Dict, List
from .template import load_template
from .engine import VideoJob, instantiate
from .sinks import render_video_to_mp4

def main():
    ap = argparse.ArgumentParser(description="Geometry-only sequence-to-video generator.")
    ap.add_argument("--template", required=True, help="Path to template YAML")
    ap.add_argument("--job-json", required=True, help="Job JSON: {id, sequences:{S1:[...],...}, seed?}")
    ap.add_argument("--out", required=True, help="Output mp4 path (supports {job_id} and {variant})")
    args = ap.parse_args()

    template = load_template(args.template)
    job_raw = json.loads(args.job_json)
    job = VideoJob(id=str(job_raw["id"]), sequences=job_raw["sequences"], seed=job_raw.get("seed"), meta=job_raw.get("meta"))

    instances = instantiate(template, job)
    for inst in instances:
        out_path = args.out.format(job_id=job.id, variant=inst.variant_idx)
        render_video_to_mp4(inst, out_path)
        print("Wrote", out_path)

if __name__ == "__main__":
    main()
