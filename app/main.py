from typing import List, Tuple, Dict, Optional, Any

import cv2

import argparse
import math
import numpy as np
import time
from dataclasses import dataclass

import os
from pathlib import Path

from control import (
    VehicleState,
    ControllerManager,
    PurePursuitController,
    StanleyController,
)
from planning import build_reference_path_from_points
from actuation import ServoMapper, ThrottleMapper, OutputInterface, ActuationCommand
from perception import LaneDetector  # "real" detector stub for now
from state_estimation import StateEstimator, SensorPacket
from utils import CsvLogger, CsvLoggerConfig

from app.config import (
    VehicleParams,
    ControllerParams,
    SpeedMode,
    default_limits,
    default_servo_cal,
    default_esc_cal,
)


# ============================================================
# Centerline generators
# ============================================================
"""
def make_fallback_path(length_m: float, ds: float = 0.10) -> Tuple[List[float], List[float]]:
    
    # Stress-test centerline: varying curvature + mixed frequencies + localized bumps.
    # Keeps x strictly increasing so resampling works nicely.
    
    xs: List[float] = []
    ys: List[float] = []

    length_m = float(length_m)
    ds = float(ds)
    n_points = max(2, int(math.ceil(length_m / ds)) + 1)

    for i in range(n_points):
        x = i * ds

        y = 0.55 * math.sin(2.0 * math.pi * x / 24.0)

        if x > 6.0:
            y += 0.20 * math.sin(2.0 * math.pi * (x - 6.0) / 6.0)

        if 10.0 <= x <= 16.0:
            u = (x - 10.0) / 6.0
            window = 0.5 - 0.5 * math.cos(2.0 * math.pi * u)
            y += window * (0.75 * math.sin(2.0 * math.pi * u))

        def gauss(xv: float, mu: float, sigma: float) -> float:
            return math.exp(-0.5 * ((xv - mu) / sigma) ** 2)

        y += 0.35 * gauss(x, mu=18.0, sigma=0.8)
        y -= 0.30 * gauss(x, mu=21.5, sigma=0.7)

        xs.append(x)
        ys.append(y)

    return xs, ys

"""

def make_fallback_path(length_m: float, ds: float = 0.10) -> Tuple[List[float], List[float]]:
    """
    More aggressive forward-only test path with sharper turns.
    Keeps x strictly increasing so resampling still works nicely.
    """
    xs: List[float] = []
    ys: List[float] = []

    length_m = float(length_m)
    ds = float(ds)
    n_points = max(2, int(math.ceil(length_m / ds)) + 1)

    for i in range(n_points):
        x = i * ds

        # Bigger slow wave
        y = 0.9 * math.sin(2.2 * math.pi * x / 20.0)

        # Stronger medium wave
        if x > 5.0:
            y += 0.55 * math.sin(1.2 * math.pi * (x - 5.0) / 5.0)

        # Sharper localized chicane
        if 9.0 <= x <= 43.0:
            u = (x - 9.0) / 8.0
            window = 0.5 - 0.5 * math.cos(2.0 * math.pi * u)
            y += window * (0.6 * math.sin(3.5 * math.pi * u))

        # A couple stronger bumps
        def gauss(xv: float, mu: float, sigma: float) -> float:
            return math.exp(-0.5 * ((xv - mu) / sigma) ** 2)

        y += 0.55 * gauss(x, mu=19.0, sigma=0.7)
        y -= 0.55 * gauss(x, mu=22.0, sigma=0.7)

        xs.append(x)
        ys.append(y)

    return xs, ys
    
def make_repeating_wavy_path(
    length_m: float,
    ds: float = 0.10,
    repeat_length_m: float = 60.0,
) -> Tuple[List[float], List[float]]:
    """
    Smooth forward path whose shape repeats every repeat_length_m.
    Useful for long stress tests without sharp seams.
    """
    xs: List[float] = []
    ys: List[float] = []

    length_m = float(length_m)
    ds = float(ds)
    L = float(repeat_length_m)

    n_points = max(2, int(math.ceil(length_m / ds)) + 1)

    for i in range(n_points):
        x = i * ds

        y = (
            6.0 * math.sin(2.0 * math.pi * x / L) +
            3.6 * math.sin(4.0 * math.pi * x / L) +
            2.4 * math.sin(8.0 * math.pi * x / L) +
            1.4 * math.sin(16.0 * math.pi * x / L)
        )

        xs.append(x)
        ys.append(y)

    return xs, ys

def _linspace_points(p0: Tuple[float, float], p1: Tuple[float, float], ds: float) -> List[Tuple[float, float]]:
    """Points along a straight segment, excluding the final endpoint (to avoid duplicates)."""
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L <= 1e-9:
        return []

    n = max(1, int(math.floor(L / ds)))
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        t = (i * ds) / L
        pts.append((x0 + t * dx, y0 + t * dy))
    return pts


def _arc_points(center: Tuple[float, float], r: float, ang0: float, ang1: float, ds: float) -> List[Tuple[float, float]]:
    """
    Points along an arc from ang0->ang1 (radians), excluding final endpoint (to avoid duplicates).
    Direction is determined by ang1-ang0 sign.
    """
    cx, cy = center
    r = float(r)
    if r <= 1e-9:
        return []

    dtheta = ds / r
    total = ang1 - ang0
    if abs(total) <= 1e-9:
        return []

    step = dtheta if total > 0 else -dtheta
    n = max(1, int(math.floor(abs(total) / abs(step))))
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        th = ang0 + i * step
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def make_rounded_rectangle_centerline(
    width_m: float,
    height_m: float,
    corner_radius_m: float,
    ds: float = 0.10,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[List[float], List[float]]:
    """
    Closed-loop rounded rectangle centered near origin_xy (bottom-left at origin_xy).
    Corners are true quarter-circle arcs of radius corner_radius_m.
    """
    W = float(width_m)
    H = float(height_m)
    r = float(corner_radius_m)
    ds = float(ds)

    if W <= 0 or H <= 0:
        raise ValueError("width_m and height_m must be > 0")
    r_max = 0.5 * min(W, H) - 1e-6
    r = max(0.0, min(r, r_max))

    ox, oy = origin_xy

    # Define key points (rectangle with rounded corners)
    # Start at bottom edge, just after bottom-left corner arc
    p_start = (ox + r, oy)

    # Straight segments endpoints (tangent points to arcs)
    p_b0 = (ox + r, oy)
    p_b1 = (ox + W - r, oy)

    p_r0 = (ox + W, oy + r)
    p_r1 = (ox + W, oy + H - r)

    p_t0 = (ox + W - r, oy + H)
    p_t1 = (ox + r, oy + H)

    p_l0 = (ox, oy + H - r)
    p_l1 = (ox, oy + r)

    # Arc centers
    c_br = (ox + W - r, oy + r)       # bottom-right
    c_tr = (ox + W - r, oy + H - r)   # top-right
    c_tl = (ox + r, oy + H - r)       # top-left
    c_bl = (ox + r, oy + r)           # bottom-left

    pts: List[Tuple[float, float]] = []

    # Bottom straight
    pts += _linspace_points(p_b0, p_b1, ds)

    # Bottom-right arc: -90° -> 0°
    if r > 0:
        pts += _arc_points(c_br, r, -math.pi/2, 0.0, ds)

    # Right straight
    pts += _linspace_points(p_r0, p_r1, ds)

    # Top-right arc: 0° -> 90°
    if r > 0:
        pts += _arc_points(c_tr, r, 0.0, math.pi/2, ds)

    # Top straight
    pts += _linspace_points(p_t0, p_t1, ds)

    # Top-left arc: 90° -> 180°
    if r > 0:
        pts += _arc_points(c_tl, r, math.pi/2, math.pi, ds)

    # Left straight
    pts += _linspace_points(p_l0, p_l1, ds)

    # Bottom-left arc: 180° -> 270°
    if r > 0:
        pts += _arc_points(c_bl, r, math.pi, 3*math.pi/2, ds)

    # Close by adding the final exact start point
    pts.append(p_start)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart Car Supreme: controller runner")

    p.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Optional saved video file to use as the frame source in real lane mode.",
    )

    p.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index for live camera input in real lane mode.",
    )

    p.add_argument(
        "--use-camera",
        action="store_true",
        help="Use a live camera instead of a saved video file in real lane mode.",
    )

    p.add_argument(
        "--show-debug",
        action="store_true",
        help="Enable lane detector debug visualization if supported.",
    )

    p.add_argument(
        "--save-debug-video",
        action="store_true",
        help="Save annotated lane detector debug video.",
    )

    p.add_argument(
        "--debug-video-out",
        type=str,
        default="lane_debug_output.mp4",
        help="Output path for saved debug video.",
    )

    p.add_argument(
        "--save-raw-video",
        action="store_true",
        help="Save raw camera/video frames.",
    )

    p.add_argument(
        "--save-failure-frames",
        action="store_true",
        help="Save raw/debug PNGs on bad or failsafe frames.",
    )

    p.add_argument(
        "--debug-video-fps",
        type=float,
        default=20.0,
        help="FPS for saved debug/raw video files.",
    )

    p.add_argument(
        "--debug-dir",
        type=str,
        default="logs",
        help="Directory for saved debug artifacts.",
    )

    # Output / CAN
    p.add_argument(
        "--output-mode",
        choices=["print", "can"],
        default="print",
        help="Actuation output backend",
    )
    p.add_argument(
        "--can-channel",
        type=str,
        default="can0",
        help="SocketCAN channel name",
    )
    p.add_argument(
        "--can-bustype",
        type=str,
        default="socketcan",
        help="CAN interface type",
    )
    p.add_argument(
        "--can-id-arm",
        type=lambda x: int(x, 0),
        default=0x200,
        help="CAN ID for arm/disarm message",
    )
    p.add_argument(
        "--can-id-ctrl",
        type=lambda x: int(x, 0),
        default=0x201,
        help="CAN ID for steering/throttle control message",
    )
    p.add_argument(
        "--can-auto-arm",
        action="store_true",
        help="Automatically arm the Arduino PWM node on startup",
    )

    p.add_argument(
        "--fallback-mode",
        choices=["path", "hold_and_stop"],
        default="hold_and_stop",
        help="Fallback behavior when lane detection fails",
    )

    p.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="How long to run simulation (seconds). If omitted, runs forever.",
    )

    # Lane source
    p.add_argument(
        "--lane-source",
        choices=["sim", "real"],
        default="sim",
        help="Lane detection source: simulated (sim) or real perception (real)",
    )

    p.add_argument(
        "--state-mode",
        choices=["auto", "camera_only", "bicycle_sim", "dead_reckoning"],
        default="auto",
        help="State estimator mode. 'auto' uses bicycle_sim for sim lanes and camera_only for real lanes.",
    )

    p.add_argument(
        "--repeat-seconds",
        type=float,
        default=60.0,
        help="For wavy_repeat, path shape repeats every this many seconds of travel.",
    )

    p.add_argument(
        "--sim-speed-mps",
        type=float,
        default=1.0,
        help="Nominal simulated vehicle speed used by the estimator.",
    )

    # Target-point lookahead used for logging/plotting (global ref/tgt sampling)
    p.add_argument(
        "--tgt-lookahead-m",
        type=float,
        default=2.0,
        help="Lookahead distance (m) for logging/plotting target point on the GLOBAL path",
    )
    p.add_argument(
        "--tgt-ds-m",
        type=float,
        default=0.10,
        help="Sampling ds (m) used when converting lookahead meters -> indices for tgt_i",
    )

    # Track shape
    p.add_argument(
        "--track",
        choices=["wavy", "wavy_repeat", "rectangle"],
        default="rectangle",
        help="Which synthetic centerline to use in SIM mode",
    )
    p.add_argument("--rect-w", type=float, default=8.0, help="Rectangle width (m)")
    p.add_argument("--rect-h", type=float, default=5.0, help="Rectangle height (m)")
    p.add_argument("--rect-r", type=float, default=1.0, help="Corner radius (m)")
    p.add_argument("--rect-ds", type=float, default=0.10, help="Rectangle sampling ds (m)")

    p.add_argument("--controller", choices=["pp", "stanley"], default="pp", help="Controller type")
    p.add_argument("--dt", type=float, default=0.05, help="Loop timestep seconds (default 0.05 => 20Hz)")

    # Speed/actuation behavior
    p.add_argument("--speed-mode", choices=["fixed_pwm", "accel_command"], default=None, help="Override SpeedMode.mode")
    p.add_argument("--motor-enabled", action="store_true", help="Enable motor output (otherwise safe throttle)")
    p.add_argument("--fixed-throttle", type=int, default=None, help="Override fixed throttle PWM (us)")
    p.add_argument("--safe-throttle", type=int, default=None, help="Override safe throttle PWM (us)")

    # Failsafe knobs
    p.add_argument("--conf-min", type=float, default=0.60, help="Lane confidence minimum before failsafe")
    p.add_argument("--max-bad-frames", type=int, default=3, help="Bad frames before failsafe triggers")

    return p.parse_args()


# ============================================================
# Helpers
# ============================================================

def _valid_detection(det) -> bool:
    if det is None:
        return False

    xs = getattr(det, "centerline_xs", None)
    ys = getattr(det, "centerline_ys", None)

    if xs is None or ys is None:
        return False
    if len(xs) < 2 or len(ys) < 2:
        return False
    if len(xs) != len(ys):
        return False

    return True


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def _nearest_path_index(path, x: float, y: float) -> Optional[int]:
    """Return index of closest point on the path to (x,y)."""
    if path is None or not getattr(path, "xs", None) or not getattr(path, "ys", None):
        return None
    xs = np.asarray(path.xs, dtype=float)
    ys = np.asarray(path.ys, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return None
    d2 = (xs - float(x)) ** 2 + (ys - float(y)) ** 2
    return int(np.argmin(d2))


# ============================================================
# Simulated lane detector (2D-safe, works on loops)
# ============================================================

@dataclass
class _SimLaneDetection:
    centerline_xs: List[float]
    centerline_ys: List[float]
    confidence: float


class _SimLaneDetector2D:
    """
    Produces a forward-looking slice of a GLOBAL path based on nearest index to (x,y).
    Works for any 2D path (loops, rectangles, etc). Wraps around for closed loops.
    """
    def __init__(
        self,
        global_path,
        horizon_m: float = 8.0,
        ds_m: float = 0.10,
        noise_std: float = 0.02,
        dropout_prob: float = 0.05,
        short_horizon_prob: float = 0.10,
        min_points: int = 12,
        closed_loop: bool = True,
    ):
        self.path = global_path
        self.horizon_m = float(horizon_m)
        self.ds_m = float(ds_m)
        self.noise_std = float(noise_std)
        self.dropout_prob = float(dropout_prob)
        self.short_horizon_prob = float(short_horizon_prob)
        self.min_points = int(min_points)
        self.closed_loop = bool(closed_loop)

        self.xs = np.asarray(self.path.xs, dtype=float)
        self.ys = np.asarray(self.path.ys, dtype=float)
        self.yaws = np.asarray(self.path.yaws, dtype=float)
        self.n = int(len(self.xs))

    @staticmethod
    def _world_to_local(
        x_world: float,
        y_world: float,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
    ) -> Tuple[float, float]:
        """
        Convert a world point into the vehicle-local frame.

        Local convention:
          x_local = forward
          y_local = left
        """
        dx = float(x_world) - float(vehicle_x)
        dy = float(y_world) - float(vehicle_y)

        c = math.cos(float(vehicle_yaw))
        s = math.sin(float(vehicle_yaw))

        x_local = dx * c + dy * s
        y_local = -dx * s + dy * c
        return float(x_local), float(y_local)

    def detect(
        self,
        frame=None,
        vehicle_x: float = 0.0,
        vehicle_y: float = 0.0,
        vehicle_yaw: float = 0.0,
    ) -> Optional[_SimLaneDetection]:
        if self.n < self.min_points:
            return None

        # Random dropout
        if np.random.rand() < self.dropout_prob:
            return None

        horizon = self.horizon_m
        if np.random.rand() < self.short_horizon_prob:
            horizon = max(0.5, 0.35 * horizon)

        n_ahead = max(self.min_points, int(round(horizon / self.ds_m)))

        # Find nearest index on global path to current (x,y)
        d2 = (self.xs - float(vehicle_x)) ** 2 + (self.ys - float(vehicle_y)) ** 2
        i0 = int(np.argmin(d2))

        if self.closed_loop:
            idx = (np.arange(i0, i0 + n_ahead) % self.n)
            xs = self.xs[idx].copy()
            ys = self.ys[idx].copy()
        else:
            i1 = min(self.n, i0 + n_ahead)
            xs = self.xs[i0:i1].copy()
            ys = self.ys[i0:i1].copy()

        if xs.size < self.min_points:
            return None

        local_xs: List[float] = []
        local_ys: List[float] = []

        for xw, yw in zip(xs, ys):
            xl, yl = self._world_to_local(
                x_world=float(xw),
                y_world=float(yw),
                vehicle_x=float(vehicle_x),
                vehicle_y=float(vehicle_y),
                vehicle_yaw=float(vehicle_yaw),
            )

            # keep only points ahead and within a lateral window
            MAX_LATERAL_M = 4.0

            if xl >= 0.0 and abs(yl) <= MAX_LATERAL_M:
                local_xs.append(float(xl))
                local_ys.append(float(yl))

        if len(local_xs) < self.min_points:
            return None

        pts = sorted(zip(local_xs, local_ys), key=lambda p: p[0])
        local_xs = [p[0] for p in pts]
        local_ys = [p[1] for p in pts]

        capped_xs = []
        capped_ys = []
        for x, y in zip(local_xs, local_ys):
            if x <= self.horizon_m:
                capped_xs.append(x)
                capped_ys.append(y)

        local_xs = capped_xs
        local_ys = capped_ys

        if self.noise_std > 0.0:
            local_ys = [
                float(y + np.random.normal(0.0, self.noise_std))
                for y in local_ys
            ]

        conf = 1.0 if horizon >= 0.9 * self.horizon_m else 0.6
        return _SimLaneDetection(local_xs, local_ys, float(conf))


def _sample_global_ref_and_tgt(global_path, state: VehicleState, lookahead_m: float, ds_m: float):
    """
    Stable (global) ref/tgt for logging & plotting.
    Uses global_path indices so ref_i/tgt_i are consistent across time.
    """
    ref_x = ref_y = ref_yaw_rad = e_cte_m = e_heading_rad = ""
    ref_i = ""
    tgt_x = tgt_y = ""
    tgt_i = ""

    i_near = _nearest_path_index(global_path, state.x, state.y)
    if isinstance(i_near, int) and 0 <= i_near < len(global_path.xs):
        ref_i = int(i_near)
        ref_x = float(global_path.xs[i_near])
        ref_y = float(global_path.ys[i_near])
        ref_yaw_rad = float(global_path.yaws[i_near])

        dx = state.x - ref_x
        dy = state.y - ref_y
        e_cte_m = math.hypot(dx, dy)
        e_heading_rad = _wrap_angle(ref_yaw_rad - state.yaw)

        lookahead_pts = max(1, int(round(float(lookahead_m) / float(ds_m))))
        ti = min(i_near + lookahead_pts, len(global_path.xs) - 1)
        tgt_i = int(ti)
        tgt_x = float(global_path.xs[ti])
        tgt_y = float(global_path.ys[ti])

    return {
        "ref_i": ref_i,
        "ref_x": ref_x,
        "ref_y": ref_y,
        "ref_yaw_rad": ref_yaw_rad,
        "e_cte_m": e_cte_m,
        "e_heading_rad": e_heading_rad,
        "tgt_i": tgt_i,
        "tgt_x": tgt_x,
        "tgt_y": tgt_y,
    }

def _log_centerline_points(
    centerline_logger,
    now: float,
    frame_idx,
    det,
    source: str,
    conf: float,
) -> None:
    if det is None:
        return

    xs = getattr(det, "centerline_xs", None)
    ys = getattr(det, "centerline_ys", None)

    if xs is None or ys is None:
        return
    if len(xs) < 2 or len(ys) < 2:
        return
    if len(xs) != len(ys):
        return

    for pt_idx, (x_m, y_m) in enumerate(zip(xs, ys)):
        centerline_logger.write({
            "t": now,
            "frame_idx": frame_idx,
            "pt_idx": pt_idx,
            "x_local_m": float(x_m),
            "y_local_m": float(y_m),
            "lane_conf": float(conf),
            "source": source,
        })

def _log_centerline_points_global(
    centerline_global_logger,
    now: float,
    frame_idx,
    det,
    source: str,
    conf: float,
    state: VehicleState,
) -> None:
    if det is None:
        return

    xs = getattr(det, "centerline_xs", None)
    ys = getattr(det, "centerline_ys", None)

    if xs is None or ys is None:
        return
    if len(xs) < 2 or len(ys) < 2:
        return
    if len(xs) != len(ys):
        return

    for pt_idx, (x_local_m, y_local_m) in enumerate(zip(xs, ys)):
        x_world_m, y_world_m = _local_to_world(
            x_local=float(x_local_m),
            y_local=float(y_local_m),
            state=state,
        )

        centerline_global_logger.write({
            "t": now,
            "frame_idx": frame_idx,
            "pt_idx": pt_idx,
            "x_local_m": float(x_local_m),
            "y_local_m": float(y_local_m),
            "x_world_m": float(x_world_m),
            "y_world_m": float(y_world_m),
            "vehicle_x": float(state.x),
            "vehicle_y": float(state.y),
            "vehicle_yaw": float(state.yaw),
            "lane_conf": float(conf),
            "source": source,
        })

def _local_to_world(
    x_local: float,
    y_local: float,
    state: VehicleState,
) -> Tuple[float, float]:
    """
    Convert a point from vehicle-local frame to world/global frame.

    Local convention used by the lane detector:
      x_local = forward
      y_local = left

    Vehicle state convention:
      state.x, state.y = world position
      state.yaw = world heading
    """
    c = math.cos(state.yaw)
    s = math.sin(state.yaw)

    x_world = state.x + x_local * c - y_local * s
    y_world = state.y + x_local * s + y_local * c
    return float(x_world), float(y_world)

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    vehicle = VehicleParams()
    cparams = ControllerParams()
    speed_mode = SpeedMode()
    limits = default_limits()

    # Apply CLI overrides
    if args.speed_mode is not None:
        speed_mode.mode = args.speed_mode
    if args.fixed_throttle is not None:
        speed_mode.fixed_throttle_pwm_us = args.fixed_throttle
    if args.safe_throttle is not None:
        speed_mode.safe_throttle_pwm_us = args.safe_throttle
    if args.motor_enabled:
        speed_mode.motor_enabled = True
    if args.fallback_mode is not None:
        speed_mode.fallback_mode = args.fallback_mode

    controller_mode = args.controller
    dt = float(args.dt)

    # Lane source selection
    LANE_SOURCE = args.lane_source
    USE_SIM_LANE_DETECTION = (LANE_SOURCE == "sim")
    USE_REAL_LANE_DETECTION = (LANE_SOURCE == "real")
    print(f"[INFO] Lane source: {LANE_SOURCE}")

    run_seconds = args.run_seconds
    run_forever = (run_seconds is None)
    start_time = time.time()

    CONFIDENCE_MIN = float(args.conf_min)
    MAX_BAD_FRAMES = int(args.max_bad_frames)
    bad_frames = 0

    # For SIM testing, don't use x-based horizon checks (rectangle breaks that assumption)

    HOLD_STOP_TGT_LOOKAHEAD_M = float(args.tgt_lookahead_m)
    PATH_RESAMPLE_DS = float(args.tgt_ds_m)

    if args.state_mode == "auto":
        estimator_mode = "bicycle_sim" if USE_SIM_LANE_DETECTION else "camera_only"
    else:
        estimator_mode = args.state_mode

    estimator = StateEstimator(wheelbase_m=vehicle.wheelbase_m, mode=estimator_mode)
    print(f"[INFO] Estimator mode: {estimator_mode}")

    # Build controller
    if controller_mode == "pp":
        controller = PurePursuitController(
            wheelbase_m=vehicle.wheelbase_m,
            params={
                "lookahead_min_m": cparams.pp_lookahead_min_m,
                "lookahead_max_m": cparams.pp_lookahead_max_m,
                "lookahead_k": cparams.pp_lookahead_k,
                "enable_speed_control": cparams.enable_speed_control,
                "target_speed_mps": cparams.target_speed_mps,
                "speed_kp": cparams.speed_kp,
            },
        )
    else:
        controller = StanleyController(
            params={
                "k": cparams.stanley_k,
                "softening": cparams.stanley_softening,
                "enable_speed_control": cparams.enable_speed_control,
                "target_speed_mps": cparams.target_speed_mps,
                "speed_kp": cparams.speed_kp,
            }
        )

    manager = ControllerManager(controller)

    servo_cal = default_servo_cal()
    servo = ServoMapper(servo_cal)
    throttle = ThrottleMapper(default_esc_cal())
    out = OutputInterface(
        mode=args.output_mode,
        can_channel=args.can_channel,
        can_bustype=args.can_bustype,
        can_id_arm=args.can_id_arm,
        can_id_ctrl=args.can_id_ctrl,
        auto_arm=args.can_auto_arm,
        send_neutral_on_start=True,
        neutral_steer_pwm_us=servo_cal.pwm_center_us,
        neutral_throttle_pwm_us=default_esc_cal().pwm_neutral_us,
    )

    # -----------------------------
    # Build GLOBAL track centerline
    # -----------------------------
    if args.track == "rectangle":
        base_xs, base_ys = make_rounded_rectangle_centerline(
            width_m=args.rect_w,
            height_m=args.rect_h,
            corner_radius_m=args.rect_r,
            ds=args.rect_ds,
            origin_xy=(0.0, 0.0),
        )
        print(f"[INFO] Track: rounded rectangle W={args.rect_w} H={args.rect_h} R={args.rect_r}")

    elif args.track == "wavy":
        estimated_speed = float(args.sim_speed_mps)
        if run_forever:
            length_m = 100.0
        else:
            length_m = max(15.0, estimated_speed * float(run_seconds) * 2.0)

        base_xs, base_ys = make_fallback_path(length_m, ds=0.15)
        print(f"[INFO] Track: wavy forward path length≈{length_m:.1f} m")

    elif args.track == "wavy_repeat":
        estimated_speed = float(args.sim_speed_mps)
        repeat_length_m = estimated_speed * float(args.repeat_seconds)

        if run_forever:
            length_m = max(3.0 * repeat_length_m, 180.0)
        else:
            required_len = estimated_speed * float(run_seconds) * 1.25
            length_m = max(required_len, 2.0 * repeat_length_m)

        base_xs, base_ys = make_repeating_wavy_path(
            length_m=length_m,
            ds=0.15,
            repeat_length_m=repeat_length_m,
        )
        print(
            f"[INFO] Track: repeating wavy forward path "
            f"(repeat every ~{args.repeat_seconds:.1f}s, "
            f"repeat_len≈{repeat_length_m:.1f} m, total_len≈{length_m:.1f} m)"
        )

    # Stable global path used for logging/plotting AND for sim detection
    global_path = build_reference_path_from_points(
        base_xs,
        base_ys,
        stamp=start_time,
        resample_ds=PATH_RESAMPLE_DS,
        smooth_window=1,  # keep 1 so we don't “fake round” corners; we already modeled arcs
    )

    # -----------------------------
    # Build detectors
    # -----------------------------
    sim_detector = _SimLaneDetector2D(
        global_path=global_path,
        horizon_m=8.0,
        ds_m=PATH_RESAMPLE_DS,
        noise_std=0.02,
        dropout_prob=0.05,
        short_horizon_prob=0.10,
        min_points=12,
        closed_loop=(args.track == "rectangle"),
    )
    real_detector = LaneDetector(
        show_debug=args.show_debug,
        min_pair_valid_fraction=0.22,
        max_crossed_fraction=0.35,
        max_width_std_px=140.0,
        temporal_alpha_center=0.74,
    )

    video_cap = None
    if USE_REAL_LANE_DETECTION:
        if args.use_camera:
            video_cap = cv2.VideoCapture(args.camera_index)
            if not video_cap.isOpened():
                raise RuntimeError(f"Could not open camera index: {args.camera_index}")
            print(f"[INFO] Using live camera index: {args.camera_index}")

        elif args.video_path:
            video_cap = cv2.VideoCapture(args.video_path)
            if not video_cap.isOpened():
                raise RuntimeError(f"Could not open video file: {args.video_path}")
            print(f"[INFO] Using saved video: {args.video_path}")

        else:
            raise RuntimeError(
                "Real lane detection requires either --use-camera or --video-path <file>."
            )

    print(f"Running controller={controller.name} at {1/dt:.1f} Hz")
    print(f"Speed mode: {speed_mode.mode} | motor_enabled={speed_mode.motor_enabled}")
    print(f"fixed_throttle_pwm_us={speed_mode.fixed_throttle_pwm_us} | safe_throttle_pwm_us={speed_mode.safe_throttle_pwm_us}")
    print(f"Fallback mode: {getattr(speed_mode, 'fallback_mode', 'hold_and_stop')}")
    print(f"Lane failsafe: CONFIDENCE_MIN={CONFIDENCE_MIN}, MAX_BAD_FRAMES={MAX_BAD_FRAMES}")
    print(f"[INFO] Global path points: {len(global_path.xs)}")

    logger = CsvLogger(CsvLoggerConfig(log_dir="logs", filename_prefix=controller.name, flush_every=10))
    print(f"Logging to: {logger.path}")

    centerline_logger = CsvLogger(
        CsvLoggerConfig(
            log_dir="logs",
            filename_prefix=f"{controller.name}_centerline",
            flush_every=50,
        )
    )
    print(f"Centerline logging to: {centerline_logger.path}")

    centerline_global_logger = CsvLogger(
        CsvLoggerConfig(
            log_dir="logs",
            filename_prefix=f"{controller.name}_centerline_global",
            flush_every=50,
        )
    )
    print(f"Global centerline logging to: {centerline_global_logger.path}")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    fail_dir = debug_dir / "fail_frames"
    if args.save_failure_frames:
        fail_dir.mkdir(parents=True, exist_ok=True)

    debug_video_writer = None
    raw_video_writer = None
    frame_idx = 0
    loop_idx = 0

    # --- Seed full global reference path into the CSV once (prevents diagonal "gap-connecting" lines) ---
    for i in range(len(global_path.xs)):
        logger.write({
            "t": start_time,
            "controller": controller.name,

            "source": "REF_SEED",
            "fallback_mode": "",
            "used_fallback": "",

            "ref_i": i,
            "ref_x": float(global_path.xs[i]),
            "ref_y": float(global_path.ys[i]),

            # keep these blank so they don't look like normal timesteps
            "tgt_i": "",
            "tgt_x": "",
            "tgt_y": "",

            "x": "",
            "y": "",
            "yaw": "",
            "v": "",

            "steer_rad": "",
            "accel_cmd": "",
            "steer_pwm_us": "",
            "throttle_pwm_us": "",

            "lane_conf": "",
            "failsafe": "",
            "bad_frames": "",
            "cmd_valid": "",
            "cmd_reason": "ref_seed",

            "fit_mode": "",
            "left_fit_ok": "",
            "right_fit_ok": "",
            "lane_seg_count": "",
            "center_pt_count": "",
            "lane_pair_valid": "",
            "lane_width_min_px": "",
            "lane_width_max_px": "",
            "lane_width_mean_px": "",
            "lane_width_std_px": "",
            "lane_width_reason": "",

            # optional debug slots
            "Ld": "",
            "cte": "",
            "heading_err": "",
            "cte_term": "",
            "kappa": "",
            "ctrl_tgt_i": "",
        })

    last_steer_pwm = servo_cal.pwm_center_us

    # Temporary nominal sensor packet.
    # In camera-only mode, speed_mps is just used for controller lookahead math.
    nominal_speed_mps = 0.8 if USE_REAL_LANE_DETECTION else 1.0
    sensors = SensorPacket(yaw=0.0, speed_mps=nominal_speed_mps)

    print(f"[INFO] Nominal controller speed: {sensors.speed_mps:.2f} m/s")

    last_debug_print_t = 0.0

    last_valid_tgt = {"tgt_x": "", "tgt_y": "", "tgt_i": ""}

    try:
        while True:
            now = time.time()
            frame = None
            loop_idx += 1

            # Optional timing diagnostics
            loop_t0 = now
            perception_ms = ""
            control_ms = ""
            loop_ms = ""

            if (not run_forever) and (run_seconds is not None) and ((now - start_time) >= run_seconds):
                print(f"\n[EXIT] Reached run limit of {run_seconds} seconds.")
                break

            state: VehicleState = estimator.peek()
            state_for_detection = state

            # ---- Perception ----
            det = None
            if USE_SIM_LANE_DETECTION:
                det = sim_detector.detect(
                    frame=None,
                    vehicle_x=state.x,
                    vehicle_y=state.y,
                    vehicle_yaw=state.yaw,
                )
            elif USE_REAL_LANE_DETECTION:
                try:
                    if video_cap is None:
                        raise RuntimeError("No camera/video source is configured for real lane detection.")

                    ok, frame = video_cap.read()

                    # If using a saved video, loop when it ends
                    if not ok and args.video_path is not None:
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = video_cap.read()

                    if not ok or frame is None:
                        raise RuntimeError("Could not read frame from camera/video source.")

                    perception_t0 = time.time()
                    det = real_detector.detect(frame=frame)
                    perception_ms = (time.time() - perception_t0) * 1000.0

                    frame_idx += 1

                    raw_h, raw_w = frame.shape[:2]

                    if args.save_raw_video and raw_video_writer is None:
                        raw_path = str(debug_dir / f"{controller.name}_raw.avi")
                        raw_video_writer = cv2.VideoWriter(
                            raw_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.debug_video_fps),
                            (raw_w, raw_h),
                        )
                        print(f"[INFO] Saving raw video to: {raw_path}")

                    debug_frame = getattr(det, "debug_frame", None)

                    if args.save_debug_video and debug_video_writer is None and debug_frame is not None:
                        dbg_h, dbg_w = debug_frame.shape[:2]
                        dbg_path = str(debug_dir / Path(args.debug_video_out).name)
                        debug_video_writer = cv2.VideoWriter(
                            dbg_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.debug_video_fps),
                            (dbg_w, dbg_h),
                        )
                        print(f"[INFO] Saving debug video to: {dbg_path}")

                    if raw_video_writer is not None:
                        raw_video_writer.write(frame)

                    if debug_video_writer is not None and debug_frame is not None:
                        debug_video_writer.write(debug_frame)

                    if args.show_debug:
                        cv2.imshow("camera_feed", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC
                            print("\n[EXIT] ESC pressed. Shutting down cleanly...")
                            break

                except Exception as e:
                    det = None
                    if (now - last_debug_print_t) > 1.0:
                        print(f"[PERCEPTION ERROR] {e}")
                        last_debug_print_t = now

            conf = float(getattr(det, "confidence", 0.0) or 0.0)

            det_debug = getattr(det, "debug_meta", None)
            if not isinstance(det_debug, dict):
                det_debug = {}

            # Keep only simple scalar-like debug values for CSV logging
            det_debug_csv = {}
            for k, v in det_debug.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    det_debug_csv[k] = v
                else:
                    det_debug_csv[k] = str(v)

            debug_meta = getattr(det, "debug_meta", {}) if det is not None else {}
            det_mode = debug_meta.get("mode", "real" if USE_REAL_LANE_DETECTION else "sim")

            if conf < CONFIDENCE_MIN:
                bad_frames += 1
            else:
                bad_frames = 0

            failsafe_active = bad_frames >= MAX_BAD_FRAMES

            should_save_failure = (
                args.save_failure_frames
                and USE_REAL_LANE_DETECTION
                and (
                    det is None
                    or conf < CONFIDENCE_MIN
                    or failsafe_active
                )
            )

            if should_save_failure:
                stamp_ms = int(now * 1000)
                if frame is not None:
                    cv2.imwrite(str(fail_dir / f"raw_{stamp_ms}_{frame_idx:06d}.png"), frame)

                det_debug_frame = getattr(det, "debug_frame", None) if det is not None else None
                if det_debug_frame is not None:
                    cv2.imwrite(str(fail_dir / f"debug_{stamp_ms}_{frame_idx:06d}.png"), det_debug_frame)

            detection_ok = _valid_detection(det) and (not failsafe_active)

            # In SIM mode: just ensure we have enough points (no x-horizon checks)
            if USE_SIM_LANE_DETECTION and detection_ok:
                if len(det.centerline_xs) < 12:
                    detection_ok = False

            fallback_mode = getattr(speed_mode, "fallback_mode", "hold_and_stop")

            if USE_REAL_LANE_DETECTION and estimator_mode == "camera_only":
                # In real camera-only mode, do not substitute a fake global path.
                fallback_mode = "hold_and_stop"

            used_fallback = int(not detection_ok)

            if detection_ok:
                source = "DETECTION"
            elif fallback_mode == "path":
                source = "FALLBACK_PATH"
            else:
                source = "HOLD_AND_STOP"

            # ---- HOLD + STOP fallback ----
            if source == "HOLD_AND_STOP":
                steer_pwm = int(last_steer_pwm)

                thr_pwm = int(speed_mode.safe_throttle_pwm_us)
                thr_pwm = int(max(throttle.cal.pwm_min_us, min(throttle.cal.pwm_max_us, thr_pwm)))

                sensors_stop = SensorPacket(yaw=0.0, speed_mps=0.0)
                state = estimator.update(sensors_stop, dt, steer_rad=None)

                global_pts = _sample_global_ref_and_tgt(
                    global_path,
                    state,
                    lookahead_m=HOLD_STOP_TGT_LOOKAHEAD_M,
                    ds_m=PATH_RESAMPLE_DS,
                )

                if global_pts["tgt_x"] != "" and global_pts["tgt_y"] != "":
                    last_valid_tgt = {"tgt_x": global_pts["tgt_x"], "tgt_y": global_pts["tgt_y"], "tgt_i": global_pts["tgt_i"]}
                else:
                    global_pts.update(last_valid_tgt)

                loop_ms = (time.time() - loop_t0) * 1000.0

                logger.write({
                    "t": now,
                    "controller": controller.name,

                    "source": source,
                    "fallback_mode": fallback_mode,
                    "used_fallback": used_fallback,

                    "x": state.x,
                    "y": state.y,
                    "yaw": state.yaw,
                    "v": state.v,

                    "steer_rad": "",
                    "accel_cmd": "",
                    "steer_pwm_us": steer_pwm,
                    "throttle_pwm_us": thr_pwm,

                    "lane_conf": conf,
                    "failsafe": int(failsafe_active),
                    "bad_frames": bad_frames,

                    "frame_idx": frame_idx if USE_REAL_LANE_DETECTION else loop_idx,
                    "camera_ok": int(USE_REAL_LANE_DETECTION and det is not None),
                    "det_mode": det_mode,
                    "left_fit_ok": debug_meta.get("left_fit_ok", ""),
                    "right_fit_ok": debug_meta.get("right_fit_ok", ""),
                    "lane_seg_count": debug_meta.get("lane_seg_count", ""),
                    "center_pt_count": debug_meta.get("center_pt_count", ""),
                    "roi_y0": debug_meta.get("roi_y0", ""),
                    "frame_width": debug_meta.get("frame_width", ""),
                    "frame_height": debug_meta.get("frame_height", ""),
                    "perception_error": debug_meta.get("error", ""),

                    "cmd_valid": 0,
                    "cmd_reason": "fallback_hold_and_stop",

                    "perception_ms": perception_ms,
                    "control_ms": control_ms,
                    "loop_ms": loop_ms,

                    "ref_x": global_pts["ref_x"],
                    "ref_y": global_pts["ref_y"],
                    "ref_i": global_pts["ref_i"],
                    "ref_yaw_rad": global_pts["ref_yaw_rad"],
                    "e_cte_m": global_pts["e_cte_m"],
                    "e_heading_rad": global_pts["e_heading_rad"],

                    "tgt_x": global_pts["tgt_x"],
                    "tgt_y": global_pts["tgt_y"],
                    "tgt_i": global_pts["tgt_i"],

                    "Ld": "",
                    "cte": "",
                    "heading_err": "",
                    "cte_term": "",
                    "kappa": "",

                    **det_debug_csv,
                })

                if (now - last_debug_print_t) > 1.0:
                    print(f"[FALLBACK hold_and_stop] conf={conf:.2f} bad_frames={bad_frames} -> SAFE throttle + hold steering")
                    last_debug_print_t = now

                out.send(ActuationCommand(steer_pwm_us=steer_pwm, throttle_pwm_us=thr_pwm))
                time.sleep(dt)
                continue

            # ---- CONTROL path ----
            if source == "DETECTION":
                xs, ys = det.centerline_xs, det.centerline_ys
            else:
                # fallback_mode == "path": use GLOBAL track
                xs, ys = list(global_path.xs), list(global_path.ys)

            if (now - last_debug_print_t) > 1.0:
                print(f"[DEBUG] source={source} conf={conf:.2f} raw_n={len(xs)}")
                last_debug_print_t = now

            path = build_reference_path_from_points(xs, ys, stamp=now, resample_ds=PATH_RESAMPLE_DS, smooth_window=1)

            control_state = state
            if source == "DETECTION":
                control_state = VehicleState(
                    x=0.0,
                    y=0.0,
                    yaw=0.0,
                    v=state.v,
                    stamp=state.stamp,
                )

            control_t0 = time.time()
            cmd = manager.update(control_state, path, limits, dt)
            control_ms = (time.time() - control_t0) * 1000.0

            debug = cmd.debug or {}
            ctrl_tgt_i = debug.get("tgt_i", None)

            steer_pwm = servo.steer_to_pwm(cmd.steer)
            last_steer_pwm = steer_pwm

            if source == "DETECTION":
                _log_centerline_points(
                    centerline_logger=centerline_logger,
                    now=now,
                    frame_idx=frame_idx if USE_REAL_LANE_DETECTION else loop_idx,
                    det=det,
                    source=source,
                    conf=conf,
                )

                _log_centerline_points_global(
                    centerline_global_logger=centerline_global_logger,
                    now=now,
                    frame_idx=frame_idx if USE_REAL_LANE_DETECTION else loop_idx,
                    det=det,
                    source=source,
                    conf=conf,
                    state=state_for_detection,
                )

            state = estimator.update(sensors, dt, steer_rad=float(cmd.steer))

            # Throttle
            if (not speed_mode.motor_enabled) or failsafe_active:
                thr_pwm = speed_mode.safe_throttle_pwm_us
            else:
                if speed_mode.mode == "fixed_pwm":
                    thr_pwm = speed_mode.fixed_throttle_pwm_us
                elif speed_mode.mode == "accel_command":
                    thr_pwm = throttle.accel_to_pwm(cmd.accel)
                else:
                    thr_pwm = speed_mode.safe_throttle_pwm_us

            thr_pwm = int(max(throttle.cal.pwm_min_us, min(throttle.cal.pwm_max_us, thr_pwm)))

            # Stable global ref/target for logging/plotting
            global_pts = _sample_global_ref_and_tgt(
                global_path,
                state,
                lookahead_m=HOLD_STOP_TGT_LOOKAHEAD_M,
                ds_m=PATH_RESAMPLE_DS,
            )

            cmd_reason = getattr(cmd, "reason", "")
            if used_fallback and source == "FALLBACK_PATH":
                cmd_reason = "fallback_path_substitution"

            loop_ms = (time.time() - loop_t0) * 1000.0

            logger.write({
                "t": now,
                "controller": controller.name,

                "source": source,
                "fallback_mode": fallback_mode,
                "used_fallback": used_fallback,

                "ref_x": global_pts["ref_x"],
                "ref_y": global_pts["ref_y"],
                "ref_i": global_pts["ref_i"],
                "ref_yaw_rad": global_pts["ref_yaw_rad"],
                "e_cte_m": global_pts["e_cte_m"],
                "e_heading_rad": global_pts["e_heading_rad"],

                "tgt_x": global_pts["tgt_x"],
                "tgt_y": global_pts["tgt_y"],
                "tgt_i": global_pts["tgt_i"],

                "x": state.x,
                "y": state.y,
                "yaw": state.yaw,
                "v": state.v,

                "steer_rad": cmd.steer,
                "accel_cmd": cmd.accel,
                "steer_pwm_us": steer_pwm,
                "throttle_pwm_us": thr_pwm,

                "lane_conf": conf,
                "failsafe": int(failsafe_active),
                "bad_frames": bad_frames,

                "frame_idx": frame_idx if USE_REAL_LANE_DETECTION else loop_idx,
                "camera_ok": int(USE_REAL_LANE_DETECTION and det is not None),
                "det_mode": det_mode,
                "left_fit_ok": debug_meta.get("left_fit_ok", ""),
                "right_fit_ok": debug_meta.get("right_fit_ok", ""),
                "lane_seg_count": debug_meta.get("lane_seg_count", ""),
                "center_pt_count": debug_meta.get("center_pt_count", ""),
                "roi_y0": debug_meta.get("roi_y0", ""),
                "frame_width": debug_meta.get("frame_width", ""),
                "frame_height": debug_meta.get("frame_height", ""),
                "perception_error": debug_meta.get("error", ""),

                "perception_ms": perception_ms,
                "control_ms": control_ms,
                "loop_ms": loop_ms,

                "cmd_valid": int(getattr(cmd, "valid", True)),
                "cmd_reason": cmd_reason,

                "Ld": debug.get("Ld", ""),
                "cte": debug.get("cte", ""),
                "heading_err": debug.get("heading_err", ""),
                "cte_term": debug.get("cte_term", ""),
                "kappa": debug.get("kappa", ""),
                "ctrl_tgt_i": ctrl_tgt_i if isinstance(ctrl_tgt_i, int) else "",

                **det_debug_csv,
            })

            out.send(ActuationCommand(steer_pwm_us=steer_pwm, throttle_pwm_us=thr_pwm))
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C received. Shutting down cleanly...")

    finally:
        try:
            out.close()
        except Exception as e:
            print(f"[EXIT] OutputInterface close warning: {e}")

        if video_cap is not None:
            video_cap.release()

        try:
            if debug_video_writer is not None:
                debug_video_writer.release()
        except Exception as e:
            print(f"[EXIT] Debug VideoWriter release warning: {e}")

        try:
            if raw_video_writer is not None:
                raw_video_writer.release()
        except Exception as e:
            print(f"[EXIT] Raw VideoWriter release warning: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        logger.close()
        print("[EXIT] Logger closed.")

        try:
            centerline_logger.close()
        except Exception as e:
            print(f"[EXIT] Centerline logger close warning: {e}")

        try:
            centerline_global_logger.close()
        except Exception as e:
            print(f"[EXIT] Global centerline logger close warning: {e}")

if __name__ == "__main__":
    main()