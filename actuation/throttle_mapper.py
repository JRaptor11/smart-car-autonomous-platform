

from dataclasses import dataclass

from control.geometry import clamp


@dataclass
class EscCal:
    """
    Calibration for an ESC controlled by PWM.
    Typical: 1000us reverse/brake, 1500us neutral, 2000us forward.
    Some ESCs are forward-only: then pwm_min_us may still exist but acts as brake.
    """
    pwm_min_us: int
    pwm_neutral_us: int
    pwm_max_us: int


class ThrottleMapper:
    """
    Map either accel command or normalized throttle into ESC PWM.

    Recommended baseline approach:
    - controller returns accel in m/s^2 (positive -> speed up, negative -> slow)
    - mapper turns that into a small delta around neutral
    """

    def __init__(self, cal: EscCal):
        self.cal = cal

        # Tunables (you can make these config-driven later)
        self.accel_deadband = 0.05          # m/s^2
        self.accel_to_pwm_gain = 120.0           # pwm_us per (m/s^2) (tune)
        self.max_pwm_step_per_call = 40     # limits jerk in PWM changes (tune)

        self._last_pwm = cal.pwm_neutral_us

    def reset(self) -> None:
        self._last_pwm = self.cal.pwm_neutral_us

    def neutral_pwm(self) -> int:
        """Return neutral/safe PWM for this ESC."""
        return int(self.cal.pwm_neutral_us)

    def clamp_pwm(self, pwm_us: int) -> int:
        """Clamp any PWM value to the ESC's allowed range."""
        return int(clamp(int(pwm_us), self.cal.pwm_min_us, self.cal.pwm_max_us))

    def accel_to_pwm(self, accel_mps2: float) -> int:
        a = accel_mps2

        # Deadband to avoid twitching
        if abs(a) < self.accel_deadband:
            target_pwm = self.cal.pwm_neutral_us
        else:
            target_pwm = self.cal.pwm_neutral_us + a * self.accel_to_pwm_gain

        # Clamp to ESC range
        target_pwm = clamp(target_pwm, self.cal.pwm_min_us, self.cal.pwm_max_us)

        # Limit step change (simple jerk limiting)
        delta = target_pwm - self._last_pwm
        if delta > self.max_pwm_step_per_call:
            target_pwm = self._last_pwm + self.max_pwm_step_per_call
        elif delta < -self.max_pwm_step_per_call:
            target_pwm = self._last_pwm - self.max_pwm_step_per_call

        self._last_pwm = int(round(target_pwm))
        return int(self._last_pwm)

    def normalized_throttle_to_pwm(self, throttle: float) -> int:
        """
        throttle in [-1, 1]
          -1 = full reverse/brake (min)
           0 = neutral
          +1 = full forward (max)
        """
        t = clamp(throttle, -1.0, 1.0)
        if t >= 0:
            pwm = self.cal.pwm_neutral_us + t * (self.cal.pwm_max_us - self.cal.pwm_neutral_us)
        else:
            pwm = self.cal.pwm_neutral_us + t * (self.cal.pwm_neutral_us - self.cal.pwm_min_us)

        return int(round(clamp(pwm, self.cal.pwm_min_us, self.cal.pwm_max_us)))
