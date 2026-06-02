#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools.yolo

YOLO-based pedestrian detection + tracking + crossing filtering.

What this module does
---------------------
1. Runs YOLO tracking over a sequence of frames.
2. Keeps only pedestrian tracks that *cross the image center horizontally*.
3. For each frame, projects the (crossing) pedestrian bounding boxes into a
   fixed 6-cell ROI grid and returns per-cell activations.

The output of :meth:`Yolo.yolo_track` is designed to be written as a CSV with
columns:
    time_stamp_ms, frame_id, cell_0 ... cell_5

This file is refactored to:
- support arbitrary dataset keys (not just PIE / MoreSMIRK)
- be import-friendly for notebooks (no sys.exit on errors)
- avoid the hard dependency on `ultralytics` at import time
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.attention_grid import roi_cell_compute
from utils.helper import infer_frame_id
from tools.sfc import calculate_morton


@dataclass
class CrossingFilterConfig:
    """Config controlling what counts as a 'crossing' track."""

    image_width: Optional[int] = None
    # Track must move at least this many pixels in x.
    yolo_track_threshold: float = 0.0


@dataclass
class ActivationConfig:
    """Config controlling the 6-cell activation computation."""

    roi_width: int = 0
    roi_height: int = 0
    # Threshold on overlap ratio [0..1] to convert to 0/1.
    sfc_bounding: float = 0.0


class Yolo:
    """YOLO tracker + ROI activation extractor."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        dataset: str,
        yolo_model_path: Optional[str] = None,
        fps: Optional[float] = None,
        person_class_id: int = 0,
        save_visualizations: bool = False,
        visualization_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.dataset = dataset
        self.verbose = verbose

        if "General" not in config:
            raise KeyError("config must contain a 'General' section")
        if "Dataset" not in config:
            raise KeyError("config must contain a 'Dataset' section")
        if dataset not in config["Dataset"]:
            raise KeyError(f"Dataset '{dataset}' not found in config['Dataset']")

        self.general_cfg = config["General"]
        self.dataset_cfg = config["Dataset"][dataset]

        # --- Runtime settings
        self.person_class_id = int(self.general_cfg.get("person_class_id", person_class_id))

        self.fps = float(self.general_cfg.get("fps", fps if fps is not None else 30.0))
        if self.fps <= 0:
            raise ValueError("fps must be > 0")

        self.crossing_cfg = CrossingFilterConfig(
            image_width=self.general_cfg.get("image_width"),
            yolo_track_threshold=float(self.general_cfg.get("yolo_track_threshold", 0)),
        )

        self.activation_cfg = ActivationConfig(
            roi_width=int(self.general_cfg.get("roi_width", 0)),
            roi_height=int(self.general_cfg.get("roi_height", 0)),
            sfc_bounding=float(self.general_cfg.get("sfc_bounding", 0.0)),
        )

        if self.activation_cfg.roi_width <= 0 or self.activation_cfg.roi_height <= 0:
            raise ValueError("General.roi_width and General.roi_height must be set to positive integers")

        # --- Optional visualization
        self.save_visualizations = bool(save_visualizations)
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else None

        # --- Model
        model_path = (
            yolo_model_path
            or self.general_cfg.get("yolo_model_path")
            or "yolo11x.pt"
        )
        self.yolo_model_path = str(model_path)
        self.device = device

        # Import ultralytics lazily so notebooks can import the package even if
        # ultralytics isn't installed yet.
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise ImportError(
                "ultralytics is required to run YOLO tracking. Install it with `pip install ultralytics`."
            ) from e

        # `device` handling: ultralytics accepts device in .track(...), but also
        # supports model.to(device). We keep it simple here.
        self._yolo = YOLO(self.yolo_model_path)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def yolo_track(
        self,
        frame_paths: Sequence[str],
        *,
        seq_name: str = "sequence",
        return_morton: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """Track pedestrians, filter for crossings, compute 6-cell activations.

        Parameters
        ----------
        frame_paths:
            Ordered list of image paths.

        seq_name:
            Used only for visualization output folder naming.

        return_morton:
            If True, also compute a per-frame Morton code (optional).

        Returns
        -------
        (sfc_rows, morton_rows)
            ``sfc_rows`` is a list of dicts with keys
            ``time_stamp_ms, frame_id, cell_0..cell_5``.
            ``morton_rows`` is ``None`` unless return_morton=True.
        """

        if not frame_paths:
            return [], [] if return_morton else None

        # -----------------------------------------------------------------
        # Pass 1: run YOLO tracking and store per-track positions.
        # -----------------------------------------------------------------

        # track_id -> list of detections
        # detection tuple: (x1, y1, x2, y2, x_center, y_center, w, h, frame_idx)
        track_dict: Dict[int, List[Tuple[int, int, int, int, float, float, float, float, int]]] = {}

        for frame_idx, img_path in enumerate(frame_paths):
            results = self._yolo.track(
                img_path,
                persist=True,
                verbose=False,
                device=self.device,
            )

            if not results:
                continue

            r0 = results[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            # Convert to numpy-backed boxes.
            boxes_np = boxes.cpu().numpy()

            if getattr(boxes_np, "id", None) is None:
                # No tracker IDs -> cannot do crossing filtering.
                continue

            cls = np.asarray(boxes_np.cls)
            person_mask = cls == self.person_class_id
            if not np.any(person_mask):
                continue

            xyxy = np.asarray(boxes_np.xyxy)[person_mask]
            xywh = np.asarray(boxes_np.xywh)[person_mask]
            ids = np.asarray(boxes_np.id)[person_mask]

            for bb_xyxy, bb_xywh, tid in zip(xyxy, xywh, ids):
                try:
                    tid_int = int(tid)
                except Exception:
                    continue

                x1, y1, x2, y2 = [int(v) for v in bb_xyxy]
                x_c, y_c, w, h = [float(v) for v in bb_xywh]

                track_dict.setdefault(tid_int, []).append(
                    (x1, y1, x2, y2, x_c, y_c, w, h, frame_idx)
                )

        # -----------------------------------------------------------------
        # Determine which tracks count as "crossing".
        # -----------------------------------------------------------------
        image_width = self._resolve_image_width(frame_paths)
        threshold = float(self.crossing_cfg.yolo_track_threshold)

        crossing_detections: List[Tuple[int, int, int, int, float, float, float, float, int]] = []
        for tid, dets in track_dict.items():
            if not dets:
                continue

            center_x_all = [d[4] for d in dets]
            min_center_x = float(min(center_x_all))
            max_center_x = float(max(center_x_all))

            crosses_center = max_center_x > (image_width / 2.0) > min_center_x
            moves_enough = (max_center_x - min_center_x) > threshold

            if crosses_center and moves_enough:
                crossing_detections.extend(dets)

        # Pre-index crossing boxes by frame for fast lookup in Pass 2.
        boxes_by_frame: Dict[int, List[np.ndarray]] = {}
        for det in crossing_detections:
            x1, y1, x2, y2, *_rest, frame_idx = det
            boxes_by_frame.setdefault(int(frame_idx), []).append(np.array([x1, y1, x2, y2], dtype=np.int32))

        # -----------------------------------------------------------------
        # Pass 2: compute per-frame 6-cell activations for crossing pedestrians.
        # -----------------------------------------------------------------

        roi_coord_all = roi_cell_compute(
            self.dataset_cfg,
            self.activation_cfg.roi_width,
            self.activation_cfg.roi_height,
        )
        n_cells = int(roi_coord_all.shape[0])
        if n_cells != 6:
            # The pipeline assumes 6 cells. Allow other counts, but warn.
            if self.verbose:
                print(f"[WARN] ROI grid has {n_cells} cells (expected 6). Output will have {n_cells} cells.")

        dt_ms = 1000.0 / self.fps

        sfc_rows: List[Dict[str, Any]] = []
        morton_rows: List[Dict[str, Any]] = []

        for frame_idx, img_path in enumerate(frame_paths):
            frame_id = infer_frame_id(img_path, default=frame_idx)
            timestamp_ms = int(round(frame_idx * dt_ms))

            yolo_boxes = boxes_by_frame.get(frame_idx, [])
            if yolo_boxes:
                yolo_coord_all = np.vstack(yolo_boxes)
                sfc_input = self._roi_overlap_ratio_per_cell(roi_coord_all, yolo_coord_all)
            else:
                sfc_input = np.zeros(n_cells, dtype=np.float32)

            # Normalize -> binary activations.
            sfc_bin = (np.asarray(sfc_input) > self.activation_cfg.sfc_bounding).astype(int)

            row = {
                "time_stamp_ms": timestamp_ms,
                "frame_id": frame_id,
            }
            for c in range(n_cells):
                row[f"cell_{c}"] = int(sfc_bin[c])
            sfc_rows.append(row)

            if return_morton:
                morton_rows.append(
                    {"time_stamp_ms": timestamp_ms, "frame_id": frame_id, "morton": int(calculate_morton(sfc_bin))}
                )

            if self.save_visualizations:
                self._save_visualization(
                    img_path=img_path,
                    roi_coord_all=roi_coord_all,
                    yolo_coord_all=np.vstack(yolo_boxes) if yolo_boxes else np.empty((0, 4), dtype=np.int32),
                    seq_name=seq_name,
                    frame_id=frame_id,
                )

        return sfc_rows, (morton_rows if return_morton else None)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _resolve_image_width(self, frame_paths: Sequence[str]) -> int:
        """Get image width from config or by reading the first frame."""

        if self.crossing_cfg.image_width is not None:
            try:
                return int(self.crossing_cfg.image_width)
            except Exception:
                pass

        # Fallback: read first frame.
        import cv2

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise FileNotFoundError(f"Could not read first frame: {frame_paths[0]}")
        return int(first.shape[1])

    @staticmethod
    def _roi_overlap_ratio_per_cell(roi_coord_all: np.ndarray, yolo_coord_all: np.ndarray) -> np.ndarray:
        """Compute union overlap area ratio per ROI cell.

        Returns values in [0..1] (clipped).
        """

        if yolo_coord_all.size == 0:
            return np.zeros(int(roi_coord_all.shape[0]), dtype=np.float32)

        roi_w = int(roi_coord_all[0, 2] - roi_coord_all[0, 0])
        roi_h = int(roi_coord_all[0, 3] - roi_coord_all[0, 1])
        cell_area = float(max(roi_w * roi_h, 1))

        ratios = np.zeros(int(roi_coord_all.shape[0]), dtype=np.float32)

        # For each cell, compute the union area of overlaps between the ROI rect
        # and all YOLO rects.
        for i, roi in enumerate(roi_coord_all):
            overlaps = []
            for yolo in yolo_coord_all:
                _area, x1, y1, x2, y2 = Yolo.find_overlap_for_two_bbox(roi, yolo)
                if _area <= 0:
                    continue
                overlaps.append((int(x1), int(y1), int(x2), int(y2)))

            if not overlaps:
                ratios[i] = 0.0
                continue

            union_area = Yolo._union_area_in_roi(tuple(map(int, roi)), overlaps)
            ratios[i] = min(float(union_area) / cell_area, 1.0)

        return ratios

    @staticmethod
    def _union_area_in_roi(roi: Tuple[int, int, int, int], overlaps: List[Tuple[int, int, int, int]]) -> int:
        """Union area of overlap rectangles *inside* a given ROI cell.

        This uses a small mask of the ROI size (roi_height x roi_width) so it is
        much cheaper than masking the full frame.
        """

        x0, y0, x1, y1 = roi
        width = max(x1 - x0, 1)
        height = max(y1 - y0, 1)

        mask = np.zeros((height, width), dtype=np.uint8)
        for ox0, oy0, ox1, oy1 in overlaps:
            # Convert global coords to ROI-local coords.
            lx0 = max(ox0 - x0, 0)
            ly0 = max(oy0 - y0, 0)
            lx1 = min(ox1 - x0, width)
            ly1 = min(oy1 - y0, height)
            if lx1 <= lx0 or ly1 <= ly0:
                continue
            mask[ly0:ly1, lx0:lx1] = 1

        return int(np.count_nonzero(mask))

    def _save_visualization(
        self,
        *,
        img_path: str,
        roi_coord_all: np.ndarray,
        yolo_coord_all: np.ndarray,
        seq_name: str,
        frame_id: int,
    ) -> None:
        """Save debug visualization (ROI grid + YOLO bboxes)."""

        if self.visualization_dir is None:
            # Default to ./outputs/visual
            out_root = Path("outputs") / "visual"
        else:
            out_root = self.visualization_dir

        out_dir = out_root / self.dataset / seq_name
        out_dir.mkdir(parents=True, exist_ok=True)

        import cv2

        bgr = cv2.imread(img_path)
        if bgr is None:
            return

        # Draw ROI cells (red)
        for roi in roi_coord_all:
            x1, y1, x2, y2 = [int(v) for v in roi]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw YOLO boxes (green)
        for bb in yolo_coord_all:
            x1, y1, x2, y2 = [int(v) for v in bb]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_path = out_dir / f"{frame_id:06d}.png"
        cv2.imwrite(str(out_path), bgr)

    # ---------------------------------------------------------------------
    # Geometry helpers (ported from the original repo)
    # ---------------------------------------------------------------------

    @staticmethod
    def find_overlap_for_two_bbox(cord_0: Iterable[int], cord_1: Iterable[int]) -> Tuple[int, int, int, int, int]:
        """Overlap between 2 bboxes in xyxy format."""

        c0 = list(map(int, cord_0))
        c1 = list(map(int, cord_1))

        x_left = max(c0[0], c1[0])
        x_right = min(c0[2], c1[2])
        y_top = max(c0[1], c1[1])
        y_bottom = min(c0[3], c1[3])

        if x_right < x_left or y_bottom < y_top:
            return 0, 0, 0, 0, 0

        overlap_size = int((x_right - x_left) * (y_bottom - y_top))
        return overlap_size, int(x_left), int(y_top), int(x_right), int(y_bottom)
