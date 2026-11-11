"""The last centimetres at home: move until the ultrasonic ahead and to the right read what they read at the start.

Near the start the cameras see only far tags (a few cm off); the echoes are good to about 1 cm.
"""
from __future__ import annotations

TRIES = 6
OK_M = 0.010
MAX_STEP_M = 0.06
MAX_ERROR_M = 0.15       # further off than this it is not the same thing as at the start: that direction is left alone
SPEED = 0.09


def read_echoes(nav) -> tuple:
    out = []
    for pan, wait in ((0.0, 0.4), (-90.0, 0.5)):
        nav.eyes.point(pan)
        nav.hold(wait, "parking: measuring")
        out.append(nav.eyes.echo(7))
    nav.eyes.point(0.0)
    return tuple(out)


def park(nav, home_echoes: tuple) -> dict:
    result = dict(measurements=0, errors_cm=None)
    for attempt in range(1, TRIES + 1):
        now = read_echoes(nav)
        errors = [None if a is None or b is None or abs(a - b) > MAX_ERROR_M else a - b for a, b in zip(now, home_echoes)]
        result.update(measurements=attempt, errors_cm=[None if e is None else e * 100 for e in errors])
        nav.rec.event("parking echoes against the start", ahead_cm=result["errors_cm"][0], right_cm=result["errors_cm"][1])
        if all(e is None or abs(e) <= OK_M for e in errors):
            break
        forward = 0.0 if errors[0] is None else max(-MAX_STEP_M, min(MAX_STEP_M, errors[0]))      # further than at the start: forward
        left = 0.0 if errors[1] is None else max(-MAX_STEP_M, min(MAX_STEP_M, -errors[1]))         # the wall further away: to the right
        length = (forward ** 2 + left ** 2) ** 0.5
        nav.nudge(SPEED * forward / length, SPEED * left / length, length / SPEED, "parking: moving")
    return result
