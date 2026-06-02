#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils.attention_grid

The project uses a fixed *6-cell* grid overlaid on the image (e.g. a zebra-crossing area).
Each cell is represented by a rectangle (x1, y1, x2, y2) in pixel coordinates.

Only :func:`roi_cell_compute` is required for the YOLO pedestrian crossing pipeline.
The older :func:`roi_values` helper (for attention heatmaps) is kept for backwards
compatibility.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def roi_values(heatmap: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Compute mean values inside each ROI cell for a 1-channel heatmap.

    Notes
    -----
    This is not used by the YOLO crossing pipeline, but it is kept because it
    existed in the original repository.
    """

    # Import lazily to keep the module lightweight when used from notebooks.
    import cv2

    roi_left_top_all = np.array(config["attention_grid"]["grid_left_top_coord"]).transpose()
    roi_right_all = roi_left_top_all[0] + config["General"]["roi_width"]
    roi_bottom_all = roi_left_top_all[1] + config["General"]["roi_height"]

    if any(i > config["mlnet_input_size"][0] for i in roi_right_all) or any(
        i > config["mlnet_input_size"][1] for i in roi_bottom_all
    ):
        raise ValueError("The attention_grid definition is beyond the image size!")

    heatmap = cv2.resize(heatmap, [640, 480], interpolation=cv2.INTER_LINEAR)
    roi_mean_values = np.zeros(len(roi_right_all), dtype=np.float32)
    for i in range(len(roi_right_all)):
        roi_mean_values[i] = np.mean(
            heatmap[
                roi_left_top_all[1][i] : roi_bottom_all[i],
                roi_left_top_all[0][i] : roi_right_all[i],
            ]
        )

    return roi_mean_values


def roi_cell_compute(roi_grid_cfg: Dict[str, Any], roi_width: int, roi_height: int) -> np.ndarray:
    """Convert a grid config to per-cell rectangles.

    Parameters
    ----------
    roi_grid_cfg:
        Dataset config section containing the key ``grid_left_top_coord``.
        Example::

            {
              "grid_left_top_coord": [[x0, y0], [x1, y1], ...]
            }

    roi_width / roi_height:
        Cell size in pixels.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_cells, 4)`` with integer coordinates ``[x1, y1, x2, y2]``.
    """

    if "grid_left_top_coord" not in roi_grid_cfg:
        raise KeyError(
            "Dataset config must contain 'grid_left_top_coord' to build the 6-cell ROI grid."
        )

    start = np.asarray(roi_grid_cfg["grid_left_top_coord"], dtype=np.int32)
    if start.ndim != 2 or start.shape[1] != 2:
        raise ValueError(
            "'grid_left_top_coord' must be a list/array of [x, y] points, one per cell."
        )

    end = start + np.asarray([roi_width, roi_height], dtype=np.int32)
    cell_coord_all = np.concatenate([start, end], axis=1)

    return cell_coord_all
