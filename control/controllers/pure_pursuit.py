from typing import Optional
import math

from ..base import BaseController
from ..geometry import clamp, nearest_path_index, rotate_into_vehicle_frame
from ..types import VehicleState, ReferencePath, ControlLimits, ControlCommand


class PurePursuitController(BaseController):
    name = "pure_pursuit"

    def __init__(self, wheelbase_m: float, params: Optional[dict] = None):
        super().__init__(params)
        self.L = wheelbase_m

        # Lookahead parameters
        self.params.setdefault("lookahead_min_m", 0.4)
        self.params.setdefault("lookahead_max_m", 1.2)

        # Curvature sensitivity (NEW)
        self.params.setdefault("curvature_gain", 5.0)

        # Speed control (DISABLED by default for camera-only setup)
        self.params.setdefault("enable_speed_control", False)
        self.params.setdefault("target_speed_mps", 1.2)
        self.params.setdefault("speed_kp", 1.0)

    # =========================================================
    # Curvature Estimation (NEW)
    # =========================================================
    def _estimate_curvature(self, xs, ys) -> float:
        if len(xs) < 3:
            return 0.0

        x1, y1 = xs[0], ys[0]
        x2, y2 = xs[1], ys[1]
        x3, y3 = xs[2], ys[2]

        # Triangle area
        area = abs(
            x1 * (y2 - y3) +
            x2 * (y3 - y1) +
            x3 * (y1 - y2)
        ) / 2.0

        a = math.hypot(x2 - x1, y2 - y1)
        b = math.hypot(x3 - x2, y3 - y2)
        c = math.hypot(x3 - x1, y3 - y1)

        if a * b * c == 0:
            return 0.0

        curvature = 4.0 * area / (a * b * c)
        return curvature

    # =========================================================
    # Main Controller
    # =========================================================
    def compute(
        self,
        state: VehicleState,
        path: ReferencePath,
        limits: ControlLimits,
        dt: float,
        prev_cmd=None,
    ) -> ControlCommand:

        if len(path.xs) < 2:
            return ControlCommand(steer=0.0, accel=0.0, valid=False, reason="path_too_short")

        i_near = nearest_path_index(state.x, state.y, path.xs, path.ys)

        # =====================================================
        # Curvature-Based Lookahead (REPLACES velocity-based)
        # =====================================================
        curvature = self._estimate_curvature(path.xs, path.ys)

        gain = float(self.params["curvature_gain"])
        Ld = self.params["lookahead_max_m"] / (1.0 + gain * curvature)

        Ld = clamp(
            Ld,
            self.params["lookahead_min_m"],
            self.params["lookahead_max_m"],
        )

        # =====================================================
        # Target point selection
        # =====================================================
        tgt_i = i_near
        accum = 0.0

        for j in range(i_near, len(path.xs) - 1):
            dx = path.xs[j + 1] - path.xs[j]
            dy = path.ys[j + 1] - path.ys[j]
            seg = math.hypot(dx, dy)

            accum += seg
            tgt_i = j + 1

            if accum >= Ld:
                break

        tx, ty = path.xs[tgt_i], path.ys[tgt_i]

        dx = tx - state.x
        dy = ty - state.y

        x_fwd, y_left = rotate_into_vehicle_frame(dx, dy, state.yaw)

        # =====================================================
        # Safety check
        # =====================================================
        if x_fwd <= 1e-3:
            return ControlCommand(
                steer=0.0,
                accel=0.0,
                valid=False,
                reason="target_behind_vehicle",
                debug={
                    "i_near": i_near,
                    "tgt_i": tgt_i,
                    "Ld": Ld,
                    "curvature": curvature,
                    "x_fwd": x_fwd,
                    "y_left": y_left,
                },
            )

        # =====================================================
        # Steering
        # =====================================================
        kappa = 2.0 * y_left / (Ld * Ld)
        steer = math.atan(self.L * kappa)

        # =====================================================
        # Speed control (optional)
        # =====================================================
        if bool(self.params.get("enable_speed_control", False)):
            v_ref = float(self.params["target_speed_mps"])
            kp = float(self.params["speed_kp"])
            accel = kp * (v_ref - state.v)
        else:
            accel = 0.0

        return ControlCommand(
            steer=steer,
            accel=accel,
            valid=True,
            debug={
                "i_near": i_near,
                "tgt_i": tgt_i,
                "Ld": Ld,
                "curvature": curvature,
                "x_fwd": x_fwd,
                "y_left": y_left,
                "kappa": kappa,
                "speed_control_enabled": bool(self.params.get("enable_speed_control", False)),
            },
        )