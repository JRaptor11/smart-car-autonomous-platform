

from dataclasses import dataclass

from control.types import ControlLimits
from actuation import ServoCal, EscCal


@dataclass
class SpeedMode:
    """
    Camera-only friendly speed control.
    - fixed_pwm: send a constant throttle PWM (recommended for now)
    - accel_command: map controller accel -> PWM (use once you have real speed feedback)
    """
    mode: str = "fixed_pwm"  # "fixed_pwm" or "accel_command"

    # Safety toggle: start with motor disabled until you're ready
    motor_enabled: bool = False

    # What we send when motor is disabled (neutral is usually 1500)
    safe_throttle_pwm_us: int = 1500

    # What we send when motor is enabled in fixed mode (tune on your car)
    fixed_throttle_pwm_us: int = 1550

    fallback_mode: str = "path"  # "path" or "hold_and_stop"

@dataclass
class VehicleParams:
    wheelbase_m: float = 0.552  # meters (update if needed)


@dataclass
class ControllerParams:
    # Speed control inside controllers (disable for camera-only)
    enable_speed_control: bool = False

    # Pure Pursuit defaults
    pp_lookahead_min_m: float = 0.3
    pp_lookahead_max_m: float = 1.2
    pp_lookahead_k: float = 0.3

    # If enable_speed_control=True later:
    target_speed_mps: float = 1.2
    speed_kp: float = 1.0

    # Stanley defaults
    stanley_k: float = 1.0
    stanley_softening: float = 0.2


def default_limits() -> ControlLimits:
    """
    Control-layer safety limits (what the controller is allowed to command).
    These CAN be tighter than the hardware range for stability.
    """
    return ControlLimits(
        steer_min=-0.45,
        steer_max=0.45,
        steer_rate_max=1.2,  # rad/s
        accel_min=-2.0,
        accel_max=2.0,
    )


def default_servo_cal() -> ServoCal:
    """
    Hardware calibration for steering servo (PWM microseconds).
    Adjust pwm_* to match your servo/Arduino output.
    steer_*_rad should represent the physical steering range you want to map.
    """
    return ServoCal(
        steer_min_rad=-0.45,
        steer_max_rad=0.45,
        pwm_min_us=1000,
        pwm_center_us=1500,
        pwm_max_us=2000,
    )


def default_esc_cal() -> EscCal:
    """
    Hardware calibration for ESC (PWM microseconds).
    """
    return EscCal(
        pwm_min_us=1000,
        pwm_neutral_us=1500,
        pwm_max_us=2000,
    )
