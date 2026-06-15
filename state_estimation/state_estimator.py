from typing import Optional, List, Tuple, Dict

import math
import time
from dataclasses import dataclass
from control.types import VehicleState


@dataclass
class SensorPacket:
    # Replace later with real sensor fields (imu_yaw, wheel_speed, etc.)
    yaw: float = 0.0
    speed_mps: float = 1.0


class StateEstimator:
    """
    State estimator / simulator.

    Modes:
    - "dead_reckoning":
        uses sensor yaw + speed, integrates x forward only
    - "bicycle_sim":
        integrates x/y/yaw using a kinematic bicycle model
    - "camera_only":
        keeps the vehicle in a local control frame at the origin
        and only updates v from the sensor packet

    Later you can replace this with real fusion (IMU + encoders + etc.).
    """

    def __init__(self, wheelbase_m: float = 0.552, mode: str = "dead_reckoning"):
        self.L = float(wheelbase_m)
        self.mode = str(mode)  # "dead_reckoning", "bicycle_sim", or "camera_only"

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 1.0

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 1.0

    def set_mode(self, mode: str) -> None:
        self.mode = str(mode)

    def peek(self) -> VehicleState:
        """
        Return the current state without advancing time.
        Use this before computing control so we don't double-integrate.
        """
        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            v=self.v,
            stamp=time.time(),
        )

    def update(
        self,
        sensors: SensorPacket,
        dt: float,
        steer_rad: Optional[float] = None,
    ) -> VehicleState:
        """
        Update internal state.

        Args:
            sensors: SensorPacket with yaw and speed_mps
            dt: timestep seconds
            steer_rad: steering angle at wheels (radians)

        Returns:
            VehicleState
        """
        now = time.time()
        dt = max(0.0, float(dt))

        # Always update speed from the packet
        self.v = max(0.0, float(sensors.speed_mps))

        if self.mode == "camera_only":
            # Keep the controller's vehicle frame pinned to the car.
            # This is the temporary camera-only behavior:
            # the path is local to the car, so the state stays local too.
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0

        elif self.mode == "bicycle_sim":
            # Kinematic bicycle model
            delta = float(steer_rad or 0.0)

            # Prevent extreme tan blow-ups
            max_abs = 1.2  # ~68 degrees
            delta = max(-max_abs, min(max_abs, delta))

            yaw_dot = (self.v / self.L) * math.tan(delta)

            # Use current yaw for x/y integration BEFORE updating yaw
            self.x += self.v * math.cos(self.yaw) * dt
            self.y += self.v * math.sin(self.yaw) * dt
            self.yaw += yaw_dot * dt

        else:
            # Dead-reckoning stub
            self.yaw = float(sensors.yaw)
            self.x += self.v * dt
            # y unchanged

        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            v=self.v,
            stamp=now,
        )