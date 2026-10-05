#!/usr/bin/env python3
"""Webots controller for Freenove4WD.proto.

Ultrasonic owns centimetres (same rule as the real cruise). PWM 900 is stall.
Put this controller folder next to your foam-wall world project, or open
webots/worlds/freenove_robot_only.wbt and copy the robot into the arena world.
"""

from controller import Robot

PWM_STALL = 900
PWM_MAX = 4095
OMEGA_MAX = 8.0
STOP_M = 0.18
CREEP_M = 0.32
TURN_PWM = 900
FWD_PWM = 1200


def pwm_to_omega(pwm: float) -> float:
    if abs(pwm) < PWM_STALL:
        return 0.0
    return max(-OMEGA_MAX, min(OMEGA_MAX, (pwm / PWM_MAX) * OMEGA_MAX))


class SurroundCruise:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        names = ("motor_fl", "motor_rl", "motor_fr", "motor_rr")
        self.motors = [self.robot.getDevice(n) for n in names]
        for motor in self.motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.pan = self.robot.getDevice("servo_pan")
        self.tilt = self.robot.getDevice("servo_tilt")
        self.pan.setPosition(0.0)
        self.tilt.setPosition(0.0)
        self.sonic = self.robot.getDevice("sonic")
        self.sonic.enable(self.timestep)
        self.front = self.robot.getDevice("cam_front")
        self.front.enable(self.timestep)

    def set_pwm(self, d1, d2, d3, d4):
        for motor, pwm in zip(self.motors, (d1, d2, d3, d4)):
            motor.setVelocity(pwm_to_omega(pwm))

    def run(self):
        while self.robot.step(self.timestep) != -1:
            dist_m = self.sonic.getValue()
            if dist_m >= CREEP_M:
                self.set_pwm(FWD_PWM, FWD_PWM, FWD_PWM, FWD_PWM)
            elif dist_m >= STOP_M:
                creep = max(PWM_STALL, FWD_PWM // 2)
                self.set_pwm(creep, creep, creep, creep)
            else:
                self.set_pwm(-TURN_PWM, -TURN_PWM, TURN_PWM, TURN_PWM)


if __name__ == "__main__":
    SurroundCruise().run()
