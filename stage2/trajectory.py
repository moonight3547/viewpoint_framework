"""V3.3 rho-first resolution and safe h_cross propagation."""
from dataclasses import dataclass
import numpy as np
from viewpoint_framework.view_space import circular_interval_contains


@dataclass
class ColumnPrior:
    azimuth_deg: float
    rho: float | None
    rho_source: str
    h_cross: float | None
    height_source: str
    direct: bool
    observed: bool
    propagation_side: str | None = None
    propagation_failure: str | None = None

    @property
    def kind(self):
        if self.observed:
            return "observed"
        if self.rho_source == "close_loop_gap_interpolated_rho":
            return "close_loop_gap"
        return "extension"


def _offset(angle, start):
    return (float(angle) - float(start)) % 360.0


def resolve_all_radii(azimuths, observed, generation, trajectory, extension_deg):
    """Resolve the entire rho field before beginning the height sub-stage."""
    ordered = sorted({float(a) for a in azimuths},
                     key=lambda a: _offset(a, generation.start_deg))
    priors = {}
    for azimuth in ordered:
        hit = trajectory.query_segment_ray_min(azimuth)
        direct = hit.direct_intersection_count > 0 and hit.rho is not None
        priors[azimuth] = ColumnPrior(
            azimuth, hit.rho, hit.rho_source,
            hit.trajectory_cross_height if direct else None,
            "direct_segment_intersection" if direct else "unresolved",
            direct, circular_interval_contains(observed, azimuth),
        )
    supported = [p for p in priors.values() if p.observed and p.rho is not None]
    if not supported:
        raise ValueError("No observed trajectory rho boundary is available.")
    start = min(supported, key=lambda p: _offset(p.azimuth_deg, observed.start_deg))
    end = min(supported, key=lambda p: abs(
        _offset(p.azimuth_deg, observed.start_deg) - observed.span_deg))
    gap_start = observed.span_deg + float(extension_deg)
    gap_end = 360.0 - float(extension_deg)
    for prior in priors.values():
        if prior.observed:
            continue
        off = _offset(prior.azimuth_deg, observed.start_deg)
        if off <= gap_start:
            prior.rho, prior.rho_source = end.rho, "right_extension_boundary_rho"
        elif off >= gap_end:
            prior.rho, prior.rho_source = start.rho, "left_extension_boundary_rho"
        else:
            t = (off-gap_start) / max(gap_end-gap_start, 1e-10)
            prior.rho = float((1-t)*end.rho + t*start.rho)
            prior.rho_source = "close_loop_gap_interpolated_rho"
    return ordered, priors


def _position(prior, height, center, frame):
    angle = np.radians(prior.azimuth_deg)
    horizontal = np.sin(angle)*frame.x_axis + np.cos(angle)*frame.z_axis
    return center + prior.rho*horizontal + height*frame.y_axis


def propagate_cross_heights(ordered, priors, observed, center, frame, safety,
                            clearances):
    """Propagate from both observed boundaries; a failure fuses that side."""
    outside = [priors[a] for a in ordered if not priors[a].observed]
    direct = [p for p in priors.values()
              if p.observed and p.direct and p.h_cross is not None]
    if not direct:
        return priors
    left_seed = min(direct, key=lambda p: _offset(p.azimuth_deg, observed.start_deg))
    right_seed = min(direct, key=lambda p: abs(
        _offset(p.azimuth_deg, observed.start_deg)-observed.span_deg))
    left, right = [], []
    for prior in outside:
        from_right = (_offset(prior.azimuth_deg, observed.start_deg)-observed.span_deg) % 360
        from_left = _offset(observed.start_deg, prior.azimuth_deg)
        (right if from_right <= from_left else left).append(
            (min(from_right, from_left), prior))
    left.sort(key=lambda x: x[0]); right.sort(key=lambda x: x[0])
    states = {"right": [right, right_seed, True], "left": [left, left_seed, True]}
    for index in range(max(len(left), len(right))):
        for side in ("right", "left"):
            queue, previous, active = states[side]
            if index >= len(queue) or not active:
                continue
            target = queue[index][1]
            if target.rho is None:
                target.propagation_failure = "rho_unavailable"
                states[side][2] = False
                continue
            point = _position(target, previous.h_cross, center, frame)
            previous_point = _position(previous, previous.h_cross, center, frame)
            threshold = clearances[target.azimuth_deg]
            endpoint_ok, _ = safety.is_position_safe(point, threshold)
            path = safety.safe_path_fraction(previous_point, point, threshold)
            if not endpoint_ok or path.safe_fraction < 1.0-1e-9:
                target.propagation_failure = (
                    "endpoint_unsafe" if not endpoint_ok else "path_unsafe")
                states[side][2] = False
                continue
            target.h_cross = float(previous.h_cross)
            target.height_source = f"propagated_{side}"
            target.propagation_side = side
            states[side][1] = target
    return priors
