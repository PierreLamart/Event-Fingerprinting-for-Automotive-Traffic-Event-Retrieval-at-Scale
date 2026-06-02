"""End-to-end pipeline runner.

The project started as a collection of standalone scripts. This file is the
"glue" that calls them in the right order.

Pipeline stages
---------------
Depending on the selected method, the pipeline runs a subset of stages:

1. (Optional) Generate saliency heatmaps for saliency models.
2. Convert model output (heatmaps / optical flow / VLM responses) to per-cell
   *cell values* over time.
3. Convert per-frame cell values to a single *Morton code* per frame.
4. Detect the event window by finding valid Morton-code sequences.
5. (Optional) Evaluate predictions against ground truth annotations.

The heavy lifting is implemented in the individual modules so you can run and
debug each stage independently.
"""

from __future__ import annotations

import datetime
import os
import sys
from typing import List, Optional, Tuple

import click
import cv2
import matplotlib.pyplot as plt

import helper
from detector_morton import main as run_detector_morton
from evaluate import main as run_evaluate
from grid_attention import main as run_grid_attention
from grid_vlm import main as run_grid_vlm
from grid_optical_flow import main as run_grid_optical_flow
from vlm import main as run_vlm
from morton import main as run_morton


def get_total_frames(video_path: str) -> int:
    """Return the number of frames for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_dataset_info(data_path: str) -> Tuple[int, int]:
    """Return (nr_videos, nr_frames) for benchmarking."""
    nr_videos = 0
    nr_frames = 0
    for video_dir, video_id, _ in helper.traverse_videos(data_path):
        video_path = os.path.join(video_dir, f"{video_id}.avi")
        nr_videos += 1
        nr_frames += get_total_frames(video_path)
    return nr_videos, nr_frames


def save_benchmark(
    output_path: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    nr_videos: int,
    nr_frames: int,
) -> None:
    """Append a short runtime summary to ``pipeline_runtime_log.txt``."""
    duration = end_time - start_time
    sec_total = duration.total_seconds()
    sec_per_video = round(sec_total / max(nr_videos, 1), 6)
    sec_per_frame = round(sec_total / max(nr_frames, 1), 6)

    log_file_path = os.path.join(output_path, "pipeline_runtime_log.txt")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"Pipeline Run: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Start Time: {start_time}\n")
        log_file.write(f"End Time: {end_time}\n")
        log_file.write(f"Total Runtime: {duration}\n")
        log_file.write(f"Total nr. videos: {nr_videos}\n")
        log_file.write(f"Total nr. frames: {nr_frames}\n")
        log_file.write(f"Seconds per video: {sec_per_video}\n")
        log_file.write(f"Seconds per frame: {sec_per_frame}\n")
        log_file.write("-" * 50 + "\n")


def _configure_matplotlib_for_pdf() -> None:
    """Use embedded fonts for high-quality vector export."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


@click.command()
@click.option(
    "--data-path",
    "-d",
    type=click.Path(exists=True, file_okay=False),
    prompt="Where do you have the input videos?",
    help="Path to the dataset directory (contains one folder per video_id).",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(file_okay=False),
    prompt="Where should the output be saved?",
    help="Directory where outputs will be written.",
)
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    prompt="Where is the configuration YML file located?",
    help="Path to the configuration YAML.",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(
        ["mlnet", "tasednet", "transalnet", "optical-flow", "llava", "gemma", "minicpmv", "chatgpt"],
        case_sensitive=False,
    ),
    prompt="What method do you want to use?",
    help="Which upstream method to use.",
)
@click.option(
    "--generate-heatmaps/--no-generate-heatmaps",
    default=None,
    help="(saliency methods) Generate heatmaps before applying the attention grid.",
)
@click.option(
    "--apply-vlm/--no-apply-vlm",
    default=None,
    help="(llava/chatgpt) Run the VLM stage. If false, only post-process existing responses.",
)
@click.option(
    "--flicker-handling/--no-flicker-handling",
    default=None,
    help="(llava/chatgpt) Apply flickering-handling when post-processing responses.",
)
@click.option("--annotations-path", type=click.Path(dir_okay=False), help="Path to ground truth YAML.")
@click.option("--cpu", is_flag=True, help="Force CPU (for stages that support it).")
def main(
    data_path: str,
    output_path: str,
    config_path: str,
    method: str,
    generate_heatmaps: Optional[bool],
    apply_vlm: Optional[bool],
    flicker_handling: Optional[bool],
    annotations_path: Optional[str],
    cpu: bool,
) -> None:
    """Run the end-to-end pipeline."""

    method = method.lower()

    # Prompt for defaults if the user did not specify flags explicitly.
    if method in {"mlnet", "tasednet", "transalnet"} and generate_heatmaps is None:
        generate_heatmaps = click.confirm("Generate saliency heatmaps?", default=True)

    if method in {"llava", "gemma", "minicpmv", "chatgpt"}:
        if apply_vlm is None:
            apply_vlm = click.confirm(f"Run the {method} VLM stage?", default=True)
        if flicker_handling is None:
            flicker_handling = click.confirm("Use flickering-handling?", default=False)
    else:
        apply_vlm = False if apply_vlm is None else apply_vlm
        flicker_handling = False if flicker_handling is None else flicker_handling

    generate_heatmaps = bool(generate_heatmaps)
    apply_vlm = bool(apply_vlm)
    flicker_handling = bool(flicker_handling)

    # Prepare output directory.
    if os.path.isdir(output_path):
        if not click.confirm("Output path already exists. Overwrite contents?", default=False):
            click.echo("Exiting...")
            raise SystemExit(1)
    else:
        click.echo("Output path does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    _configure_matplotlib_for_pdf()

    nr_videos, nr_frames = get_dataset_info(data_path)
    start_time = datetime.datetime.now()
    click.echo(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------
    # Stage 1: saliency heatmaps
    # -------------------------
    if generate_heatmaps:
        click.echo("-" * 40)
        click.echo("Generating saliency heatmaps...")
        click.echo("-" * 40)

        if method == "mlnet":
            from saliency.MLNet.run import main as run_saliency_mlnet

            run_saliency_mlnet(data_path, output_path, config_path, cpu)
        elif method == "tasednet":
            from saliency.TASEDNet.run import main as run_saliency_tasednet

            run_saliency_tasednet(data_path, output_path, config_path, cpu)
        elif method == "transalnet":
            from saliency.TranSalNet.run import main as run_saliency_transalnet

            run_saliency_transalnet(data_path, output_path, config_path, cpu)
        else:
            raise click.ClickException(f"Heatmap generation is not supported for method '{method}'.")

    # -------------------------
    # Stage 2: cell values
    # -------------------------
    if method in {"mlnet", "tasednet", "transalnet"}:
        click.echo("-" * 40)
        click.echo("Applying attention grid...")
        click.echo("-" * 40)
        run_grid_attention(data_path, output_path, config_path)

    if apply_vlm:
        click.echo("-" * 40)
        click.echo(f"Applying {method}...")
        click.echo("-" * 40)
        run_vlm(data_path, output_path, config_path, method)

    if method in {"llava", "gemma", "minicpmv", "chatgpt"}:
        click.echo("-" * 40)
        click.echo(f"Processing {method} responses into a grid...")
        click.echo("-" * 40)
        run_grid_vlm(data_path, output_path, config_path, flicker_handling)

    if method == "optical-flow":
        click.echo("-" * 40)
        click.echo("Applying optical flow grid...")
        click.echo("-" * 40)
        run_grid_optical_flow(data_path, output_path, config_path, use_cpu=cpu)

    # -------------------------
    # Stage 3: Morton codes
    # -------------------------
    click.echo("-" * 40)
    click.echo("Generating Morton codes...")
    click.echo("-" * 40)
    run_morton(output_path)

    end_time = datetime.datetime.now()
    click.echo(f"Pipeline ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    save_benchmark(output_path, start_time, end_time, nr_videos, nr_frames)

    # -------------------------
    # Stage 4/5: detect + evaluate
    # -------------------------
    use_attention = method in {"mlnet", "tasednet", "transalnet", "llava", "gemma", "minicpmv", "chatgpt"}

    click.echo("-" * 40)
    click.echo(f"Running detector...")
    click.echo("-" * 40)
    run_detector_morton(output_path, config_path, use_attention)

    if annotations_path:
        click.echo("-" * 40)
        click.echo("Running evaluation...")
        click.echo("-" * 40)
        run_evaluate(output_path, annotations_path)

    click.echo("=" * 40)
    click.echo(f"Pipeline completed. Output saved to '{output_path}'.")
    click.echo("=" * 40)


if __name__ == "__main__":
    main()
