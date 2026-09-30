import json
import tempfile
from pathlib import Path

import numpy as np

from viewpoint_framework.stage3.block_partition import BlockConfig, plan_content_blocks
from viewpoint_framework.stage3 import render_output
from viewpoint_framework.stage3.render_output import save_stage3_outputs
from viewpoint_framework.stage3.types import (
    CandidateOrigin, ReferenceSelectionResult, SelectionCandidate, Stage3Result,
)
from viewpoint_framework.stage3.visibility import NullVisibilityModel
from viewpoint_framework.utils.cameras import Camera


def make_candidates(count):
    candidates = []
    for index in range(count):
        angle = 2.0 * np.pi * index / count
        position = np.array([np.cos(angle), 0.15 * np.sin(3 * angle), np.sin(angle)])
        forward = -position / np.linalg.norm(position)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
        camera = Camera(index, 500, 500, 320, 240, 640, 480,
                        np.linalg.inv(c2w), c2w)
        candidates.append(SelectionCandidate(
            candidate_id=index, camera=camera, origin=CandidateOrigin.GRID,
            grid_id=index, row=index // 20, col=index % 20,
            azimuth_deg=float(np.degrees(angle)), elevation_deg=0.0,
            observation_direction=position,
        ))
    return candidates


def test_160_targets_become_balanced_multiple_of_seven_blocks():
    plan = plan_content_blocks(
        make_candidates(160), scene_scale=2.0,
        visibility_model=NullVisibilityModel(), config=BlockConfig(mode="content"),
    )
    lengths = [len(block.candidates) for block in plan.blocks]
    assert lengths == [28, 28, 28, 28, 21, 21]
    assert sum(lengths) == 154
    assert len(plan.dropped_candidate_ids) == 6
    ids = [candidate.candidate_id for block in plan.blocks for candidate in block.candidates]
    assert len(ids) == len(set(ids)) == 154
    assert all(length % 7 == 0 for length in lengths)


def test_168_targets_form_six_blocks_of_28_without_drops():
    plan = plan_content_blocks(
        make_candidates(168), scene_scale=2.0,
        visibility_model=NullVisibilityModel(), config=BlockConfig(mode="content"),
    )
    assert [len(block.candidates) for block in plan.blocks] == [28] * 6
    assert plan.dropped_candidate_ids == []
    assert all(block.descriptor_strategy == "pose_fallback" for block in plan.blocks)


def test_strict_policy_rejects_non_multiple_of_seven():
    try:
        plan_content_blocks(
            make_candidates(15), scene_scale=2.0,
            visibility_model=NullVisibilityModel(),
            config=BlockConfig(mode="content", frame_count_policy="strict"),
        )
    except ValueError as exc:
        assert "multiple of 7" in str(exc)
    else:
        raise AssertionError("strict block policy must reject 15 targets")


def test_block_output_writes_multirow_refs_and_lengths(monkeypatch):
    candidates = make_candidates(56)
    plan = plan_content_blocks(
        candidates, scene_scale=2.0, visibility_model=NullVisibilityModel(),
        config=BlockConfig(mode="content"),
    )
    for block in plan.blocks:
        block.reference_original_indices = [block.block_id, block.block_id + 10]
    ordered = [candidate for block in plan.blocks for candidate in block.candidates]
    result = Stage3Result(
        selected_candidates=ordered,
        selected_cameras=[candidate.camera for candidate in ordered],
        reference_result=ReferenceSelectionResult("block_summary", [], []),
        all_candidates=candidates,
        holes=[],
        hole_views=[],
        block_plan=plan,
    )

    class FakeRenderer:
        def render_rgb(self, *_args, **_kwargs):
            return np.zeros((1, 1, 3), dtype=np.float32)

    monkeypatch.setattr(render_output, "_write_rgb_png", lambda *_args: None)
    tests_dir = Path(__file__).parent
    with tempfile.TemporaryDirectory(dir=tests_dir) as directory:
        paths = save_stage3_outputs(
            result, FakeRenderer(), directory, debug_mode=False,
            stage2_grid_candidates=candidates,
        )
        refs = json.loads(Path(paths["traj_refs"]).read_text())
        lengths = json.loads(Path(paths["traj_lens"]).read_text())
        blocks = json.loads(Path(paths["pano_blocks"]).read_text())
    assert lengths == [28, 28]
    assert refs == [[0, 10], [1, 11]]
    assert sum(lengths) == len(ordered)
    assert [block["start_index"] for block in blocks["blocks"]] == [0, 28]
