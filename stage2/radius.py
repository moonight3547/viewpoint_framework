"""V3.3 center guard and soft over-nominal radius policy."""
from dataclasses import dataclass


@dataclass
class RadiusTargets:
    probe: float
    nominal: float
    extension: float | None
    over_nominal: bool
    extension_allowed: bool


def center_guard(local_clearance, renderer_near_plane, ratio=1.0):
    return max(float(ratio) * float(local_clearance), float(renderer_near_plane))


def classify_inside_signed_radius(raw_signed_radius, guard):
    value, guard = float(raw_signed_radius), float(guard)
    if value >= guard:
        return "same_side", value
    if value <= -guard:
        return "crossing", value
    return "center_guard", guard


def resolve_radius_targets(probe_radius, nominal_cap, *, hole_detected=False):
    probe, cap = abs(float(probe_radius)), float(nominal_cap)
    if probe <= cap:
        return RadiusTargets(probe, probe, None, False, False)
    extension = (2.0 * cap + probe) / 3.0
    return RadiusTargets(probe, cap, None if hole_detected else extension,
                         True, not hole_detected)
