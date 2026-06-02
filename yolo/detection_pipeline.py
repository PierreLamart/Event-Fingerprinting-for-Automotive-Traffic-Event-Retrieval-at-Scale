#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detection_pipeline

Refactored entry-point for the YOLO + 6-cell ROI activation extraction.

Key refactor points
-------------------
- `make_input_list` is now **generic** and can ingest a new dataset by
  configuring `config['Dataset'][<dataset_key>]`.
- The pipeline is now usable from a Jupyter notebook via `process_dataset` or
  `process_sequence`.
- CSV output contains the requested **6D cell activations over time**:
  `time_stamp_ms, frame_id, cell_0..cell_5`.

The original script structure is kept (single file entry point) but rewritten to
be import-friendly and to avoid hard-coded dataset names.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from tools.yolo import Yolo
from utils import helper


# -----------------------------------------------------------------------------
# Dataset ingestion
# -----------------------------------------------------------------------------


def make_input_list(
    *,
    dataset: str,
    config: Dict[str, Any],
    input_override: Optional[str] = None,
) -> List[Tuple[List[str], str]]:
    """Create a list of (frame_paths, seq_name) pairs for a dataset.

    This is the method you were pointed to in the task description.

    Supported dataset specs (via config)
    ------------------------------------
    In `config['Dataset'][dataset]`, you can define any of the following:

    1) PIE split YAML (original behaviour)
       - yaml_path: relative yaml file path (under pie_splits_root)
       - pie_splits_root (optional, default './datasets/pie_splits')
       - dataset_root (optional, default './datasets/pie')
       - include_end (optional, default False)

    2) A single directory of frames (one sequence)
       - input_path: path to a folder containing frames
       - recursive (optional, default False)

    3) A root directory with multiple subfolders (each subfolder is a sequence)
       - input_root: path to folder containing per-sequence subfolders
       - recursive (optional, default False)

    4) A glob pattern
       - frame_glob: e.g. '/data/seq01/*.jpg'

    Notes
    -----
    - `input_override` can override `input_path` / `input_root` / `frame_glob`
      from the config, which is convenient in notebooks.
    """

    if "Dataset" not in config or dataset not in config["Dataset"]:
        raise KeyError(f"Dataset '{dataset}' not found in config['Dataset']")

    ds_cfg = config["Dataset"][dataset]

    # --- Case 1: PIE YAML split
    if dataset.lower() == "pie" or "yaml_path" in ds_cfg:
        yaml_path = ds_cfg.get("yaml_path")
        if not yaml_path:
            raise KeyError("PIE dataset config must contain 'yaml_path'")

        pie_splits_root = ds_cfg.get("pie_splits_root", "./datasets/pie_splits")
        input_yml = Path(pie_splits_root) / str(yaml_path)

        import yaml  # local import: optional dependency

        with input_yml.open("r", encoding="utf-8") as f:
            input_list = yaml.safe_load(f)

        dataset_root = ds_cfg.get("dataset_root", "./datasets/pie")
        include_end = bool(ds_cfg.get("include_end", False))

        sequences: List[Tuple[List[str], str]] = []
        for seq in input_list:
            frame_paths, seq_stub = helper.make_pie_png_list(
                seq, dataset_root=dataset_root, include_end=include_end
            )
            seq_name = f"{Path(yaml_path).stem}/{seq_stub}"
            sequences.append((frame_paths, seq_name))
        return sequences

    # --- Case 2/3/4: generic datasets

    # Highest priority: explicit override.
    input_spec = input_override

    # Else try config keys in a sensible order.
    input_spec = input_spec or ds_cfg.get("frame_glob") or ds_cfg.get("input_path") or ds_cfg.get("input_root")
    if input_spec is None:
        raise KeyError(
            "Dataset config must include one of: 'yaml_path', 'frame_glob', 'input_path', or 'input_root'."
        )

    recursive = bool(ds_cfg.get("recursive", False))

    # 4) Glob pattern
    if any(ch in str(input_spec) for ch in ["*", "?", "["]):
        frames = sorted(glob.glob(str(input_spec)), key=lambda p: helper.natural_key(Path(p).stem))
        seq_name = ds_cfg.get("seq_name", Path(str(input_spec)).parent.name or "sequence")
        return [(frames, str(seq_name))]

    p = Path(str(input_spec))

    # If user passed a list file, read it.
    if p.is_file() and p.suffix.lower() in {".txt", ".lst"}:
        lines = helper.read_list_file(p)
        # If lines are directories, treat each as a sequence. If lines are files, treat all as one sequence.
        if all(Path(x).is_dir() for x in lines):
            out: List[Tuple[List[str], str]] = []
            for d in lines:
                frames = helper.sorted_image_files(d, recursive=recursive)
                out.append((frames, Path(d).name))
            return out
        else:
            frames = [str(Path(x)) for x in lines if helper.is_image_file(x)]
            seq_name = ds_cfg.get("seq_name", p.stem)
            return [(frames, str(seq_name))]

    # If input_spec points to a directory, auto-detect whether it's:
    #  - a single sequence folder (contains images directly), or
    #  - a dataset root (contains subfolders, each a sequence).
    if p.is_dir():
        # Fast check: does it contain any images directly?
        direct_images = [f for f in p.iterdir() if f.is_file() and helper.is_image_file(f)]
        if direct_images:
            frames = helper.sorted_image_files(p, recursive=recursive)
            seq_name = ds_cfg.get("seq_name", p.name)
            return [(frames, str(seq_name))]

        # Else treat subdirectories as sequences (if any)
        subdirs = sorted([d for d in p.iterdir() if d.is_dir()], key=lambda d: d.name)
        sequences: List[Tuple[List[str], str]] = []
        for sd in subdirs:
            frames = helper.sorted_image_files(sd, recursive=recursive)
            if frames:
                sequences.append((frames, sd.name))

        if sequences:
            return sequences

        # Final fallback: recurse and collect images under p.
        frames = helper.sorted_image_files(p, recursive=True)
        return [(frames, str(ds_cfg.get("seq_name", p.name)))]

    # If we get here, `input_spec` exists but wasn't understood.
    raise ValueError(
        f"Unsupported dataset input specification: {input_spec}. "
        "Use 'frame_glob' or point to a directory (single sequence or root with subfolders), "
        "or pass a list file (*.txt)."
    )


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------


def process_sequence(
    *,
    frame_paths: Sequence[str],
    seq_name: str,
    dataset: str,
    config: Dict[str, Any],
    output_dir: Optional[str | os.PathLike] = None,
    save_visualizations: bool = False,
    visualization_dir: Optional[str | os.PathLike] = None,
    write_morton: bool = False,
    csv_sep: str = ";",
) -> Dict[str, Any]:
    """Process one sequence and (optionally) write CSV outputs.

    Returns
    -------
    dict with keys:
        - sfc_df: pandas DataFrame
        - morton_df: pandas DataFrame or None
        - sfc_csv_path: written path or None
        - morton_csv_path: written path or None
    """

    yolo = Yolo(
        config=config,
        dataset=dataset,
        save_visualizations=save_visualizations,
        visualization_dir=visualization_dir,
    )

    sfc_rows, morton_rows = yolo.yolo_track(
        list(frame_paths),
        seq_name=seq_name,
        return_morton=write_morton,
    )

    # Build DataFrames with stable column ordering.
    if sfc_rows:
        n_cells = len([k for k in sfc_rows[0].keys() if k.startswith("cell_")])
    else:
        # Default to 6 for empty outputs.
        n_cells = 6

    sfc_cols = ["time_stamp_ms", "frame_id"] + [f"cell_{i}" for i in range(n_cells)]
    sfc_df = pd.DataFrame(sfc_rows, columns=sfc_cols)

    morton_df = None
    if write_morton:
        morton_df = pd.DataFrame(morton_rows or [], columns=["time_stamp_ms", "frame_id", "morton"])

    # Write outputs
    sfc_csv_path = None
    morton_csv_path = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        sfc_csv_path = output_dir / f"{seq_name}_sfc_input.csv"
        helper.dir_path_check(sfc_csv_path)
        sfc_df.to_csv(sfc_csv_path, sep=csv_sep, index=False)

        if write_morton and morton_df is not None:
            morton_csv_path = output_dir / f"{seq_name}_morton.csv"
            helper.dir_path_check(morton_csv_path)
            morton_df.to_csv(morton_csv_path, sep=csv_sep, index=False)

    return {
        "sfc_df": sfc_df,
        "morton_df": morton_df,
        "sfc_csv_path": str(sfc_csv_path) if sfc_csv_path is not None else None,
        "morton_csv_path": str(morton_csv_path) if morton_csv_path is not None else None,
    }


def process_dataset(
    *,
    dataset: str,
    config: Dict[str, Any],
    output_dir: Optional[str | os.PathLike] = None,
    input_override: Optional[str] = None,
    save_visualizations: bool = False,
    visualization_dir: Optional[str | os.PathLike] = None,
    write_morton: bool = False,
    csv_sep: str = ";",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Process all sequences for a dataset.

    Returns a list of per-sequence results (each result is the dict returned by
    :func:`process_sequence`).
    """

    sequences = make_input_list(dataset=dataset, config=config, input_override=input_override)
    results: List[Dict[str, Any]] = []

    for frame_paths, seq_name in sequences:
        if verbose:
            print(f"Processing {dataset}: {seq_name} ({len(frame_paths)} frames)")

        res = process_sequence(
            frame_paths=frame_paths,
            seq_name=seq_name,
            dataset=dataset,
            config=config,
            output_dir=output_dir,
            save_visualizations=save_visualizations,
            visualization_dir=visualization_dir,
            write_morton=write_morton,
            csv_sep=csv_sep,
        )
        results.append(res)

    return results


# -----------------------------------------------------------------------------
# Optional CLI wrapper (still handy when not using notebooks)
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO pedestrian crossing -> 6-cell activation CSV")
    p.add_argument("-d", "--dataset", type=str, required=True, help="Dataset key in config.json (e.g. pie, MoreSMIRK, custom)")
    p.add_argument("-c", "--config", type=str, default="config.json", help="Path to config.json")
    p.add_argument("-o", "--output_path", type=str, required=True, help="Directory to write CSV outputs")
    p.add_argument("--input", dest="input_override", type=str, default=None, help="Override dataset input (dir/glob/list file)")
    p.add_argument("--save_visualizations", action="store_true", help="Write debug images with ROI+YOLO boxes")
    p.add_argument("--visualization_dir", type=str, default=None, help="Where to write visualizations")
    p.add_argument("--write_morton", action="store_true", help="Also write Morton code CSV")
    p.add_argument("--csv_sep", type=str, default=";", help="CSV separator (default ';')")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    process_dataset(
        dataset=args.dataset,
        config=cfg,
        output_dir=args.output_path,
        input_override=args.input_override,
        save_visualizations=args.save_visualizations,
        visualization_dir=args.visualization_dir,
        write_morton=args.write_morton,
        csv_sep=args.csv_sep,
        verbose=True,
    )


if __name__ == "__main__":
    main()
