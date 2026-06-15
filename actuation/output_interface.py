
from typing import Optional, List, Tuple, Dict

import struct
import time
from dataclasses import dataclass

try:
    import can
except ImportError:
    can = None


@dataclass
class ActuationCommand:
    steer_pwm_us: int
    throttle_pwm_us: int


class OutputInterface:
    """
    Hardware abstraction layer for actuation output.

    Modes:
    - "print": debug output only
    - "can": send steering/throttle PWM over CAN

    CAN protocol matches the manual keyboard script:
    - ARM  message: CAN ID 0x200, payload [0 or 1]
    - CTRL message: CAN ID 0x201, payload struct.pack("<HH", steer_us, thr_us)
    """

    def __init__(
        self,
        mode: str = "print",
        *,
        print_period_s: float = 1.0,
        can_channel: str = "can0",
        can_bustype: str = "socketcan",
        can_id_arm: int = 0x200,
        can_id_ctrl: int = 0x201,
        auto_arm: bool = False,
        send_neutral_on_start: bool = True,
        neutral_steer_pwm_us: int = 1500,
        neutral_throttle_pwm_us: int = 1500,
    ):
        self.mode = str(mode).lower()
        self._last_print_t = 0.0
        self._print_period_s = float(print_period_s)

        self.can_channel = str(can_channel)
        self.can_bustype = str(can_bustype)
        self.can_id_arm = int(can_id_arm)
        self.can_id_ctrl = int(can_id_ctrl)

        self.auto_arm = bool(auto_arm)
        self.send_neutral_on_start = bool(send_neutral_on_start)
        self.neutral_steer_pwm_us = int(neutral_steer_pwm_us)
        self.neutral_throttle_pwm_us = int(neutral_throttle_pwm_us)

        self._bus = None
        self._armed = False

        if self.mode == "can":
            if can is None:
                raise RuntimeError(
                    "python-can is not installed. Install it with: pip install python-can"
                )

            self._bus = can.interface.Bus(
                channel=self.can_channel,
                bustype=self.can_bustype,
            )

            if self.send_neutral_on_start:
                self._send_ctrl_low_level(
                    self.neutral_steer_pwm_us,
                    self.neutral_throttle_pwm_us,
                )

            self.arm(self.auto_arm)

    def arm(self, armed: bool) -> None:
        self._armed = bool(armed)

        if self.mode != "can" or self._bus is None:
            print(f"[ACTUATION] ARM={1 if armed else 0}")
            return

        msg = can.Message(
            arbitration_id=self.can_id_arm,
            is_extended_id=False,
            data=bytes([1 if armed else 0]),
        )
        self._bus.send(msg)

    def _send_ctrl_low_level(self, steer_pwm_us: int, throttle_pwm_us: int) -> None:
        if self.mode != "can" or self._bus is None:
            return

        data = struct.pack("<HH", int(steer_pwm_us), int(throttle_pwm_us))
        msg = can.Message(
            arbitration_id=self.can_id_ctrl,
            is_extended_id=False,
            data=data,
        )
        self._bus.send(msg)

    def send(self, cmd: ActuationCommand) -> None:
        now = time.time()

        if self.mode == "print":
            if now - self._last_print_t >= self._print_period_s:
                print(
                    f"[ACTUATION] steer={cmd.steer_pwm_us}us "
                    f"throttle={cmd.throttle_pwm_us}us armed={1 if self._armed else 0}"
                )
                self._last_print_t = now
            return

        if self.mode == "can":
            try:
                self._send_ctrl_low_level(cmd.steer_pwm_us, cmd.throttle_pwm_us)
            except Exception as e:
                if now - self._last_print_t >= self._print_period_s:
                    print(f"[ACTUATION ERROR] Failed to send CAN message: {e}")
                    self._last_print_t = now
                return

            if now - self._last_print_t >= self._print_period_s:
                print(
                    f"[ACTUATION CAN] steer={cmd.steer_pwm_us}us "
                    f"throttle={cmd.throttle_pwm_us}us armed={1 if self._armed else 0}"
                )
                self._last_print_t = now

    def close(self) -> None:
        try:
            if self.mode == "can" and self._bus is not None:
                try:
                    self._send_ctrl_low_level(
                        self.neutral_steer_pwm_us,
                        self.neutral_throttle_pwm_us,
                    )
                except Exception:
                    pass

                try:
                    self.arm(False)
                except Exception:
                    pass

                try:
                    self._bus.shutdown()
                except Exception:
                    pass
        finally:
            self._bus = None