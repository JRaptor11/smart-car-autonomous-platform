

import math
from typing import List, Optional, Tuple

from control.types import ReferencePath
from control.geometry import wrap_angle


def _compute_yaws(xs: List[float], ys: List[float]) -> List[float]:
    """
    Compute tangent yaw at each point.
    For point i, yaw is computed from (i -> i+1), last yaw repeats.
    """
    n = len(xs)
    if n < 2:
        return [0.0] * n

    yaws: List[float] = []
    for i in range(n - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        yaws.append(math.atan2(dy, dx))
    yaws.append(yaws[-1])
    return yaws


def _resample_by_arclength(
    xs: List[float],
    ys: List[float],
    ds: float,
) -> Tuple[List[float], List[float]]:
    """
    Resample polyline points to approximately constant arc-length spacing ds.
    Keeps the first and last points.
    """
    if len(xs) < 2 or ds <= 0:
        return xs, ys

    # Compute cumulative arc length
    s = [0.0]
    for i in range(1, len(xs)):
        s.append(s[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))

    total = s[-1]
    if total < 1e-9:
        return xs, ys

    # Target sample positions
    num = max(2, int(total / ds) + 1)
    s_targets = [i * total / (num - 1) for i in range(num)]

    # Interpolate along segments
    out_xs: List[float] = []
    out_ys: List[float] = []
    j = 0
    for st in s_targets:
        while j < len(s) - 2 and s[j + 1] < st:
            j += 1

        s0, s1 = s[j], s[j + 1]
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if abs(s1 - s0) < 1e-9:
            t = 0.0
        else:
            t = (st - s0) / (s1 - s0)

        out_xs.append(x0 + t * (x1 - x0))
        out_ys.append(y0 + t * (y1 - y0))

    return out_xs, out_ys


def _moving_average(xs: List[float], window: int) -> List[float]:
    """
    Simple moving average for smoothing 1D list.
    Uses edge padding (repeat endpoints).
    """
    if window <= 1 or len(xs) < 3:
        return xs

    w = int(window)
    pad = w // 2
    padded = [xs[0]] * pad + xs + [xs[-1]] * pad

    out: List[float] = []
    for i in range(pad, len(padded) - pad):
        out.append(sum(padded[i - pad : i + pad + 1]) / (2 * pad + 1))
    return out


def build_reference_path_from_points(
    xs: List[float],
    ys: List[float],
    stamp: float = 0.0,
    *,
    resample_ds: Optional[float] = None,
    smooth_window: int = 0,
    compute_curvature: bool = False,
) -> ReferencePath:
    """
    Convert a polyline (xs, ys) into a ReferencePath.
    Optional:
      - resample_ds: resample points to constant spacing (meters)
      - smooth_window: moving average window (odd-ish works best, e.g., 5 or 7)
      - compute_curvature: add curvature estimate (optional, not required now)
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same length")

    if len(xs) < 2:
        return ReferencePath(xs=xs, ys=ys, yaws=[0.0] * len(xs), curvatures=None, stamp=stamp)

    # Resample first (helps smoothing be consistent)
    if resample_ds is not None and resample_ds > 0:
        xs, ys = _resample_by_arclength(xs, ys, ds=resample_ds)

    # Smooth (simple, fast)
    if smooth_window and smooth_window > 1:
        xs = _moving_average(xs, smooth_window)
        ys = _moving_average(ys, smooth_window)

    # Compute yaws
    yaws = _compute_yaws(xs, ys)

    curvatures = None
    if compute_curvature and len(xs) >= 3:
        # Simple discrete curvature approximation using yaw differences / arc length
        curvatures = [0.0] * len(xs)
        for i in range(1, len(xs) - 1):
            ds1 = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            ds2 = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            ds_avg = max(1e-6, 0.5 * (ds1 + ds2))
            dyaw = wrap_angle(yaws[i + 1] - yaws[i - 1])
            curvatures[i] = dyaw / (2.0 * ds_avg)
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]

    return ReferencePath(xs=xs, ys=ys, yaws=yaws, curvatures=curvatures, stamp=stamp)
