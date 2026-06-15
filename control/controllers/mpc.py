

from ..base import BaseController
from ..types import VehicleState, ReferencePath, ControlLimits, ControlCommand


class MPCController(BaseController):
    name = "mpc"

    def compute(self, state: VehicleState, path: ReferencePath, limits: ControlLimits, dt: float, prev_cmd=None) -> ControlCommand:
        return ControlCommand(
            steer=0.0,
            accel=0.0,
            valid=False,
            reason="mpc_not_implemented",
            debug={"hint": "Drop in MPC later using the same interface"},
        )
