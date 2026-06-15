from typing import Optional, List, Tuple, Dict

import math

from ..base import BaseController
from ..geometry import nearest_path_index, rotate_into_vehicle_frame, wrap_angle
from ..types import VehicleState, ReferencePath, ControlLimits, ControlCommand


class StanleyController(BaseController):
    name = "stanley"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)

        # Lateral control params
        self.params.setdefault("k", 1.0)                 # cross-track gain
        self.params.setdefault("softening", 0.2)         # avoids blow-up at low speed

        # Longitudinal control (DISABLED by default for camera-only setup)
        self.params.setdefault("enable_speed_control", False)
        self.params.setdefault("target_speed_mps", 1.2)
        self.params.setdefault("speed_kp", 1.0)

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

        px, py = path.xs[i_near], path.ys[i_near]
        path_yaw = path.yaws[i_near]

        # Heading error
        heading_err = wrap_angle(path_yaw - state.yaw)

        # Cross-track error in vehicle frame (positive = path is left of vehicle)
        dx = px - state.x
        dy = py - state.y
        _, y_left = rotate_into_vehicle_frame(dx, dy, state.yaw)
        cte = y_left

        v = max(state.v, 0.0)
        k = float(self.params["k"])
        soft = float(self.params["softening"])

        # Stanley steering law
        cte_term = math.atan2(k * cte, (v + soft))
        steer = heading_err + cte_term

        # Speed control (only if enabled)
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
                "heading_err": heading_err,
                "cte": cte,
                "cte_term": cte_term,
                "path_yaw": path_yaw,
                "speed_control_enabled": bool(self.params.get("enable_speed_control", False)),
            },
        )
