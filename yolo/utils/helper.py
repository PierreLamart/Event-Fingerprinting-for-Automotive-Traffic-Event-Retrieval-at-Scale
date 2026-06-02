#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils.helper

Small helpers used by the pedestrian-crossing pipeline.

The refactor keeps the original PIE helper (:func:`make_pie_png_list`) and adds
more generic dataset utilities so the pipeline can ingest new datasets without
further code changes.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


_IMAGE_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def make_pie_png_list(
    seq: dict,
    *,
    dataset_root: str | os.PathLike = "./datasets/pie",
    ext: str = ".png",
    include_end: bool = False,
) -> Tuple[List[str], str]:
    """Build a list of PIE frame paths for a single sequence.

    Parameters
    ----------
    seq:
        One element from the PIE split YAML.
        Expected keys: ``id`` and ``event_window``.

    dataset_root:
        Root folder that contains PIE sequences (each sequence is a subfolder).

    ext:
        Image extension (default: ``.png``).

    include_end:
        Whether to include the last index of ``event_window``.
        The original code used Python's half-open range (end excluded). Some PIE
        split files store an exclusive end already; others store an inclusive end.
        This flag allows you to match your split convention.

    Returns
    -------
    (input_png_list, seq_name)
    """

    dir_name = str(seq["id"])
    start_frame = int(seq["event_window"][0])
    end_frame = int(seq["event_window"][-1])

    # Keep original behaviour (end excluded) unless include_end=True.
    end_range = end_frame + 1 if include_end else end_frame

    input_png_list = [
        os.path.join(str(dataset_root), dir_name, ("%03d" % i) + ext)
        for i in range(start_frame, end_range)
    ]

    seq_name = f"{dir_name}_{start_frame}_{end_frame}"
    return input_png_list, seq_name


def dir_path_check(full_path: str | os.PathLike) -> None:
    """Create parent directory for a file path if it does not exist."""

    full_path = Path(full_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)


def natural_key(s: str) -> List[object]:
    """Key for natural sorting (e.g. frame_2 before frame_10)."""

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def sorted_image_files(
    path: str | os.PathLike,
    *,
    recursive: bool = False,
    exts: Sequence[str] = _IMAGE_EXTS_DEFAULT,
) -> List[str]:
    """Collect and naturally-sort image files from a directory."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.is_file():
        # If a file is passed, just return it (caller can decide).
        return [str(p)]

    if recursive:
        files = [f for f in p.rglob("*") if f.suffix.lower() in exts]
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]

    files_sorted = sorted((str(f) for f in files), key=lambda x: natural_key(Path(x).stem))
    return files_sorted


def infer_frame_id(path: str | os.PathLike, default: int) -> int:
    """Infer frame_id from the filename.

    The original repo assumed filenames were numeric (e.g. 001.png). For new
    datasets we fall back to the frame index.
    """

    stem = Path(path).stem

    # Most specific: full stem is an int.
    try:
        return int(stem)
    except ValueError:
        pass

    # Else extract last integer group from stem.
    m = re.findall(r"\d+", stem)
    if m:
        try:
            return int(m[-1])
        except ValueError:
            return default

    return default


def is_image_file(path: str | os.PathLike, exts: Sequence[str] = _IMAGE_EXTS_DEFAULT) -> bool:
    return Path(path).suffix.lower() in set(e.lower() for e in exts)


def read_list_file(list_path: str | os.PathLike) -> List[str]:
    """Read a text file containing one path per line (blank lines ignored)."""

    p = Path(list_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    lines = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines
