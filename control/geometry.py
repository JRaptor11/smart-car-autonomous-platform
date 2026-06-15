

import math
from typing import List, Tuple


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rotate_into_vehicle_frame(dx: float, dy: float, yaw_world: float) -> Tuple[float, float]:
    """
    Convert world-frame delta (dx, dy) into vehicle frame (x_fwd, y_left).
    yaw_world is vehicle heading in world frame.
    """
    c = math.cos(-yaw_world)
    s = math.sin(-yaw_world)
    x_fwd = c * dx - s * dy
    y_left = s * dx + c * dy
    return x_fwd, y_left


def nearest_path_index(x: float, y: float, xs: List[float], ys: List[float]) -> int:
    """Return index of closest path point (naive linear search)."""
    best_i = 0
    best_d2 = float("inf")
    for i, (px, py) in enumerate(zip(xs, ys)):
        dx = px - x
        dy = py - y
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i
