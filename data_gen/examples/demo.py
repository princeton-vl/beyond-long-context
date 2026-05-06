import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidgeom import load_template
from vidgeom.engine import VideoJob, instantiate
from vidgeom.sinks import render_video_to_mp4, render_video_to_tensors

def main():
    # Conveyor demo
    tpl = load_template(os.path.join(os.path.dirname(__file__), "template_conveyor.yaml"))
    job = VideoJob(id="job_conveyor_001", sequences={"S1": list("ABBCBBABCBCBCCCCBCA")})
    instances = instantiate(tpl, job)
    for inst in instances:
        out = os.path.join(os.path.dirname(__file__), f"out_conveyor_v{inst.variant_idx}.mp4")
        print("Rendering", out)
        render_video_to_mp4(inst, out)

    # Locker demo (two sequences)
    tpl = load_template(os.path.join(os.path.dirname(__file__), "template_locker.yaml"))
    job = VideoJob(
        id="job_locker_001",
        sequences={
            "S_people": list("ABACABBBCCAA"),
            "S_items":  list("CCBAAACBBACB"),
        }
    )
    inst = instantiate(tpl, job)[0]
    out = os.path.join(os.path.dirname(__file__), "out_locker.mp4")
    print("Rendering", out)
    render_video_to_mp4(inst, out)

    # Sorting hub demo (control sequence)
    tpl = load_template(os.path.join(os.path.dirname(__file__), "template_sorting_hub.yaml"))
    job = VideoJob(
        id="job_hub_001",
        sequences={
            "S1": list("ABBCBBABCBCBCCCCBCA"),
            "S_ctrl": list("LRSLRSLRSLRSLRS"),
        }
    )
    inst = instantiate(tpl, job)[0]
    out = os.path.join(os.path.dirname(__file__), "out_sorting_hub.mp4")
    print("Rendering", out)
    render_video_to_mp4(inst, out)

    # Tensor streaming example (requires torch; commented out by default)
    # tpl = load_template(os.path.join(os.path.dirname(__file__), "template_conveyor.yaml"))
    # inst = instantiate(tpl, VideoJob(id="job_tensor_001", sequences={"S1": list("ABCABCABCABC")}))[0]
    # for batch, meta in render_video_to_tensors(inst, batch_size=16):
    #     print("Batch", batch.shape, meta["job_id"], "t0=", meta["times"][0])
    #     break

if __name__ == "__main__":
    main()
