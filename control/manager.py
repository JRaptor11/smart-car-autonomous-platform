from typing import List, Tuple, Dict, Optional, Any

from .base import BaseController
from .geometry import clamp
from .types import VehicleState, ReferencePath, ControlLimits, ControlCommand


class ControllerManager:
    """
    Wraps any controller with shared actuator safety:
    - clamps steer/accel
    - optional steering rate limit
    - fallback if controller returns invalid
    """

    def __init__(self, controller: BaseController):
        self.controller = controller
        self.last_cmd = ControlCommand(steer=0.0, accel=0.0, valid=True, reason="init")

    def set_controller(self, controller: BaseController, params: Optional[dict] = None) -> None:
        self.controller = controller
        self.controller.reset()
        if params:
            self.controller.configure(params)

    def update(self, state: VehicleState, path: ReferencePath, limits: ControlLimits, dt: float) -> ControlCommand:
        cmd = self.controller.compute(state, path, limits, dt, prev_cmd=self.last_cmd)

        if not cmd.valid:
            # Fail-safe: hold steering, stop accelerating.
            cmd.debug["manager_fallback"] = True
            cmd.steer = self.last_cmd.steer
            cmd.accel = 0.0

        # Hard limits
        cmd.steer = clamp(cmd.steer, limits.steer_min, limits.steer_max)
        cmd.accel = clamp(cmd.accel, limits.accel_min, limits.accel_max)

        # Optional global steering rate limiting
        if limits.steer_rate_max is not None and dt > 0:
            max_delta = limits.steer_rate_max * dt
            delta = cmd.steer - self.last_cmd.steer
            if delta > max_delta:
                cmd.steer = self.last_cmd.steer + max_delta
                cmd.debug["steer_rate_limited"] = True
            elif delta < -max_delta:
                cmd.steer = self.last_cmd.steer - max_delta
                cmd.debug["steer_rate_limited"] = True

        self.last_cmd = cmd
        return cmd
