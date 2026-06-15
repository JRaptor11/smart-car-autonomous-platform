

from dataclasses import dataclass

from control.geometry import clamp


@dataclass
class ServoCal:
    """
    Calibration for a standard RC steering servo controlled by PWM.
    Typical ranges: 1000-2000us, center ~1500us (but varies).
    """
    steer_min_rad: float
    steer_max_rad: float
    pwm_min_us: int
    pwm_center_us: int
    pwm_max_us: int


class ServoMapper:
    """
    Maps desired steering angle (rad) to PWM microseconds.

    Assumes steer_min_rad maps to pwm_min_us,
            0 rad maps to pwm_center_us (approximately),
            steer_max_rad maps to pwm_max_us.
    """

    def __init__(self, cal: ServoCal):
        self.cal = cal

    def steer_to_pwm(self, steer_rad: float) -> int:
        # Clamp to calibrated steering range
        steer = clamp(steer_rad, self.cal.steer_min_rad, self.cal.steer_max_rad)

        # Map piecewise around center (more accurate if asymmetry exists)
        if steer >= 0.0:
            # [0 .. steer_max] -> [center .. max]
            t = steer / (self.cal.steer_max_rad + 1e-9)
            pwm = self.cal.pwm_center_us + t * (self.cal.pwm_max_us - self.cal.pwm_center_us)
        else:
            # [steer_min .. 0] -> [min .. center]
            t = (steer - self.cal.steer_min_rad) / (0.0 - self.cal.steer_min_rad + 1e-9)
            pwm = self.cal.pwm_min_us + t * (self.cal.pwm_center_us - self.cal.pwm_min_us)

        pwm_i = int(round(pwm))
        return int(clamp(pwm_i, self.cal.pwm_min_us, self.cal.pwm_max_us))
