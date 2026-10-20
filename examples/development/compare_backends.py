#!/usr/bin/env python
"""Compare CPU and MPS throughput using the normal PyTorch backend.

Example:
    uv run --extra cpu python examples/development/compare_backends.py \\
        --output examples/development/benchmark-results/mps-comparison.json

Results are reported in MLUPS.  The timer wraps an explicit device
synchronization, so asynchronous MPS dispatch is included in the measurement.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import torch

from lettuce import Context, Simulation, StreamingStrategy
from lettuce.ext import BGKCollision, Guo, flow_by_name


DEFAULT_DEVICES = ("cpu", "mps")
DEFAULT_FLOWS = ("taylor2d", "poiseuille2d", "couette2d")
DEFAULT_RESOLUTIONS = (128, 256, 512, 1024)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"),
                        default=DEFAULT_DEVICES)
    parser.add_argument("--flows", nargs="+", choices=sorted(flow_by_name),
                        default=DEFAULT_FLOWS)
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        help="Write JSON here and CSV beside it.")
    return parser.parse_args()


def make_simulation(device, flow_name, resolution):
    """Create the same non-native simulation configured by the CLI benchmark."""
    context = Context(device=device, dtype=torch.float32, use_native=False)
    flow_class, stencil = flow_by_name[flow_name]
    flow = flow_class(context, resolution=resolution, reynolds_number=1,
                      mach_number=0.05, stencil=stencil())
    force = Guo(
        flow=flow,
        tau=flow.units.relaxation_parameter_lu,
        acceleration=flow.acceleration,
    ) if hasattr(flow, "acceleration") else None
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu,
                             force=force)
    simulation = Simulation(flow, collision, [],
                            streaming_strategy=StreamingStrategy.PRE_STREAMING)
    return context, simulation


def measure(device, flow_name, resolution, warmup_steps, steps):
    """Return end-to-end MLUPS, including queued MPS work."""
    context, simulation = make_simulation(device, flow_name, resolution)

    if warmup_steps:
        simulation(warmup_steps)
    context.synchronize()

    start = perf_counter()
    simulation(steps)
    context.synchronize()
    elapsed_seconds = perf_counter() - start
    mlups = steps * simulation.flow.rho().numel() / 1e6 / elapsed_seconds
    return mlups, elapsed_seconds


def summarise(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["device"], record["flow"], record["resolution"])].append(
            record["mlups"])

    summaries = []
    for (device, flow, resolution), samples in sorted(grouped.items()):
        summaries.append({
            "device": device,
            "flow": flow,
            "resolution": resolution,
            "runs": len(samples),
            "median_mlups": median(samples),
            "mean_mlups": mean(samples),
            "stdev_mlups": stdev(samples) if len(samples) > 1 else 0.0,
        })

    by_case = {(item["flow"], item["resolution"], item["device"]): item
               for item in summaries}
    for item in summaries:
        cpu = by_case.get((item["flow"], item["resolution"], "cpu"))
        if item["device"] == "mps" and cpu:
            item["speedup_over_cpu"] = item["median_mlups"] / cpu["median_mlups"]
        else:
            item["speedup_over_cpu"] = 1.0 if item["device"] == "cpu" else None
    return summaries


def write_results(output, records, summaries):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": records, "summary": summaries},
                                 indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {output} and {csv_path}")


def main():
    args = parse_args()
    for value, name in ((args.steps, "steps"),
                        (args.warmup_steps, "warmup-steps"),
                        (args.repeats, "repeats")):
        if value < 0 or (name != "warmup-steps" and value == 0):
            raise SystemExit(f"--{name} must be positive")

    records = []
    print("Benchmarking float32 PyTorch operations with cuda_native disabled.")
    for flow in args.flows:
        for resolution in args.resolutions:
            for repeat in range(1, args.repeats + 1):
                for device in args.devices:
                    mlups, elapsed_seconds = measure(
                        device, flow, resolution, args.warmup_steps, args.steps)
                    record = {
                        "device": device,
                        "flow": flow,
                        "resolution": resolution,
                        "repeat": repeat,
                        "mlups": mlups,
                        "elapsed_seconds": elapsed_seconds,
                    }
                    records.append(record)
                    print(f"{flow:12} {resolution:4} {device:3} "
                          f"run {repeat}: {mlups:8.2f} MLUPS")

    summaries = summarise(records)
    print("\nSummary (median MLUPS):")
    for item in summaries:
        speedup = item["speedup_over_cpu"]
        suffix = "" if speedup is None else f", {speedup:.2f}x CPU"
        print(f"{item['flow']:12} {item['resolution']:4} {item['device']:3} "
              f"{item['median_mlups']:8.2f} ± {item['stdev_mlups']:.2f}{suffix}")

    if args.output:
        write_results(args.output, records, summaries)


if __name__ == "__main__":
    main()
