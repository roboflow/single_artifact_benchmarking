#!/usr/bin/env python3
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import fire

from sab.results import load_results, pretty_print_results
from sab.runner import parse_filter

DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parent / "models")


def _filter_flag(name: str, value) -> list[str]:
    names = parse_filter(value)
    return [f"--{name}={','.join(sorted(names))}"] if names else []


def main(
    image_dir,
    annotation_file,
    buffer_time=0.0,
    models_dir=DEFAULT_MODELS_DIR,
    output_dir="benchmark_results",
    runtimes=None,
    devices=None,
    max_images=None,
    rerun=False,
):
    """Run all benchmark models and collect outputs into one list."""

    Path(output_dir).mkdir(exist_ok=True)

    flags = _filter_flag("runtimes", runtimes) + _filter_flag("devices", devices)
    if max_images is not None:
        flags.append(f"--max_images={max_images}")
    if rerun:
        flags.append("--rerun")

    scripts = sorted(glob.glob(f"{models_dir}/benchmark_*.py"))
    all_results = []

    for script in scripts:
        script_name = Path(script).stem
        output_file = f"{output_dir}/{script_name}_results.txt"

        try:
            subprocess.run(
                [sys.executable, script, image_dir, annotation_file, str(buffer_time), output_file, *flags],
                check=True,
            )

            if not os.path.exists(output_file):
                raise ValueError(f"Output file {output_file} does not exist")
            all_results.extend(load_results(output_file))

        except subprocess.CalledProcessError:
            print(f"Failed to run {script}")

    combined_file = f"{output_dir}/combined_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    pretty_print_results(all_results)


if __name__ == "__main__":
    fire.Fire(main)
