

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float          # radians, world-frame heading
    v: float            # m/s
    stamp: float        # seconds


@dataclass
class ReferencePath:
    xs: List[float]
    ys: List[float]
    yaws: List[float]                 # radians, tangent heading at each point
    curvatures: Optional[List[float]] = None
    stamp: float = 0.0


@dataclass
class ControlLimits:
    steer_min: float                  # rad
    steer_max: float                  # rad
    steer_rate_max: Optional[float]   # rad/s (None disables)
    accel_min: float                  # m/s^2
    accel_max: float                  # m/s^2


@dataclass
class ControlCommand:
    steer: float                      # rad
    accel: float                      # m/s^2
    valid: bool
    reason: str = ""
    debug: Dict = field(default_factory=dict)
