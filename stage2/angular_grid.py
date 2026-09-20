"""V3.3 angular proposal construction."""
from dataclasses import replace

from viewpoint_framework.stage1.view_domain import analyze_view_domain


def build_v33_bbox(cameras, base_bbox, profile, mode, grid_config):
    grid_config.validate()
    grid_config.up_axis = profile.coordinate_frame.y_axis.copy()
    stats = analyze_view_domain(cameras, base_bbox.observed_azimuth, mode, grid_config)
    bbox = replace(
        base_bbox,
        strategy="v3_3_eye_pitch_fov_circular",
        generation_azimuth=stats.generation_azimuth,
        generation_elevation_deg=stats.generation_elevation_range_deg,
        angular_extension_deg=stats.azimuth_extension_deg,
        notes=list(base_bbox.notes) + [
            "V3.3 E2 eye-pitch/effective-FOV proposal.",
            f"azimuth_close_loop={stats.azimuth_close_loop_applied}",
        ],
    )
    return bbox, stats
