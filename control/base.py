

from typing import Dict, Optional

from .types import VehicleState, ReferencePath, ControlLimits, ControlCommand


class BaseController:
    """
    Base interface: all controllers must implement compute().
    """

    name: str = "base"

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {}

    def reset(self) -> None:
        """Clear internal state, integrators, buffers."""
        pass

    def configure(self, params: Dict) -> None:
        """Update tunable params at runtime."""
        self.params.update(params)

    def compute(
        self,
        state: VehicleState,
        path: ReferencePath,
        limits: ControlLimits,
        dt: float,
        prev_cmd: Optional[ControlCommand] = None,
    ) -> ControlCommand:
        raise NotImplementedError
