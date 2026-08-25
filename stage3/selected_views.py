#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load the preselected captured-view set produced before Stage 1.

Expected directory layout (all files optional except selection.json for explicit
mapping):
    select_view_dir/
      selection.json              # {"selected_indices": [orig_idx, ...]}
      selected_cameras.json       # 18-D cameras, renamed in selection order
      selected_images/
        frame_0000.png ...

``selected_indices`` always refers to the original train_cameras.json / train_images
indices.  The order is preserved and is used to map selected_cameras back to the
original sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from viewpoint_framework.cameras_util import Camera, load_cameras_json
from viewpoint_framework.stage3.types import SelectedViewSet


def _find_selected_images(directory: Path) -> list[str]:
    image_dir = directory / "selected_images"
    if not image_dir.is_dir():
        return []
    paths = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
        paths.extend(image_dir.glob(suffix))
    return [str(p.resolve()) for p in sorted(set(paths))]


def load_selected_view_set(
    select_view_dir: Optional[str],
    captured_cameras: Sequence[Camera],
    *,
    fallback_count: int = 40,
) -> SelectedViewSet:
    """Load selected captured views, or deterministically fall back to ~40 frames."""
    captured_cameras = list(captured_cameras)
    n = len(captured_cameras)
    if n == 0:
        return SelectedViewSet(original_indices=[], cameras=[])

    if select_view_dir:
        root = Path(select_view_dir).expanduser().resolve()
        selection_path = root / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(f"selection.json not found: {selection_path}")
        with open(selection_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        indices = [int(x) for x in payload.get("selected_indices", [])]
        if not indices:
            raise ValueError(f"selection.json has no selected_indices: {selection_path}")
        if any(i < 0 or i >= n for i in indices):
            raise IndexError("selected_indices contains an index outside train_cameras.json")

        selected_cameras_path = root / "selected_cameras.json"
        if selected_cameras_path.is_file():
            loaded = load_cameras_json(str(selected_cameras_path))
            if len(loaded) != len(indices):
                raise ValueError(
                    "selected_cameras.json length does not match selection.json: "
                    f"{len(loaded)} vs {len(indices)}"
                )
            cameras = []
            for original_index, camera in zip(indices, loaded):
                camera.index = int(original_index)
                cameras.append(camera)
        else:
            cameras = [captured_cameras[i] for i in indices]

        return SelectedViewSet(
            original_indices=indices,
            cameras=cameras,
            source_dir=str(root),
            image_paths=_find_selected_images(root),
        )

    # Fallback only exists to keep standalone runs robust.  The actual business
    # path should normally supply --select_view_dir.
    count = max(1, min(int(fallback_count), n))
    if count == n:
        indices = list(range(n))
    else:
        import numpy as np

        indices = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, count)))
    return SelectedViewSet(
        original_indices=indices,
        cameras=[captured_cameras[i] for i in indices],
        source_dir=None,
    )


if __name__ == "__main__":
    print("stage3.selected_views: import OK")
