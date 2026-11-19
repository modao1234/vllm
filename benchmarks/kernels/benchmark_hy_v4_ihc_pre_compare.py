# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare iHC pre against a source file from a baseline vLLM checkout."""

import argparse
import csv
import importlib.util
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_pre
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _graph_us(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(16):
            fn()
    graph.replay()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 16)
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("ihc_baseline", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    set_random_seed(0)
    torch.set_num_threads(1)
    d = args.hidden_size
    dtype = getattr(torch, args.dtype)
    weight = torch.randn(8, 4 * d, device="cuda", dtype=torch.float32)
    scale = torch.randn(2, device="cuda", dtype=torch.float32) * 0.01
    base = torch.randn(8, device="cuda", dtype=torch.float32)
    print(
        f"device={current_platform.get_device_name()}; hidden={d}; hc=4; "
        f"dtype={dtype}; CUDA Graph; warm cache; {args.rounds} rounds",
        file=sys.stderr,
    )
    writer = csv.writer(sys.stdout)
    writer.writerow(["tokens", "baseline_us", "current_us", "speedup", "accuracy"])
    for n in [2**i for i in range(16)]:
        x = torch.randn(n, 4, d, device="cuda", dtype=dtype)

        old = partial(baseline.triton_ihc_pre, x, weight, scale, base, 2.0, 1e-6, 1e-5)
        new = partial(triton_ihc_pre, x, weight, scale, base, 2.0, 1e-6, 1e-5)

        expected, actual = old(), new()
        torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=1e-5)
        del expected, actual
        times = {old: [], new: []}
        for index in range(args.rounds):
            for fn in [old, new] if index % 2 == 0 else [new, old]:
                times[fn].append(_graph_us(fn))
        old_us, new_us = (statistics.median(times[fn]) for fn in (old, new))
        writer.writerow([n, old_us, new_us, old_us / new_us, "PASS"])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
