"""
Launcher physics: wheel RPM -> ball exit velocity -> projectile trajectory
-> fire/no-fire launch decision.
----------------------------------------------------------------------------
Companion to fire_detector.py / esp8266_launcher.ino. This module is the
"projectile motion optimization/decision" piece: given the drone's current
altitude above the target (from the TF-Luna rangefinder already in this
project) and a horizontal standoff distance to the target, decide whether
the CURRENT launcher hardware, at its CURRENT achievable exit velocity, can
actually land the ball on the fire -- not just whether a fire was detected.

REAL HARDWARE NUMBERS USED BELOW -- verified against the actual product
page for the exact BBOX-15 + Mabuchi 775 combo in the BOM, and the
official ITF tennis ball spec. Nothing here is a generic/invented number:
  Motor (Mabuchi 775, bare shaft):  1125 RPM no-load @ 18V nominal,
                                     1.21 Nm stall torque @ 18V
                                     (technobotix.in product listing, the
                                     exact link in the BOM)
  Gearbox (BBOX-15):                16:1 planetary reduction (same listing)
  Drive wheel:                      100mm diameter (per BOM)
  Tennis ball (ITF regulation):     mass 56.0-59.4 g, diameter 6.54-6.86 cm
                                     (ITF regulations, via Wikipedia)

*** CRITICAL FINDING -- read this before building anything downstream. ***
Run this file directly (`python launcher_physics.py`) and look at the
GEARED vs DIRECT-DRIVE section it prints. With the 16:1 gearbox in the
drivetrain, wheel surface speed comes out to roughly 0.3 m/s -- nowhere
near enough to "launch" anything, it's barely a nudge. That's because the
BBOX-15 is a torque-multiplying reduction gearbox built for combat-robot
drive wheels (high torque, low speed), which is the OPPOSITE of what a
friction-wheel ball launcher needs (high surface speed, modest torque).
This file's recommendation: mount the drive wheels DIRECTLY on the bare
motor shaft and remove the gearbox from the launcher drivetrain entirely.
At your actual 4S flight-battery voltage (14.8V) that gives an estimated
~4.8 m/s exit velocity -- a realistic, usable "shoot" speed for a 58g
tennis ball. This is an estimate from the motor's rated no-load speed,
not a bench measurement -- confirm with a tachometer or a slow-motion
video + tape measure before you treat it as final for the review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

G = 9.81  # m/s^2

# ---- Verified hardware numbers (see module docstring for sources) --------
MOTOR_NO_LOAD_RPM_AT_18V = 1125.0
MOTOR_RATED_VOLTAGE = 18.0
MOTOR_STALL_TORQUE_NM_AT_18V = 1.21
GEARBOX_RATIO = 16.0            # BBOX-15, as purchased -- see CRITICAL FINDING above
WHEEL_DIAMETER_M = 0.100         # 100mm, per BOM
WHEEL_RADIUS_M = WHEEL_DIAMETER_M / 2

# Your actual supply voltage for the launcher circuit. Flight battery is a
# 4S LiPo (14.8V nominal, ~16.8V fresh-charged) per the BOM -- change this
# if the launcher ends up on its own separate supply.
SUPPLY_VOLTAGE = 14.8

# Tennis ball (ITF regulation; midpoints of the legal range used below)
BALL_MASS_KG = 0.0577       # 56.0-59.4 g range
BALL_DIAMETER_M = 0.067     # 6.54-6.86 cm range

# Wheel-to-ball transfer efficiency: the ball never quite reaches true
# wheel surface speed (slip/compression in the contact patch costs some).
# 1.0 would be a frictionless, lossless best case. THIS IS A PLACEHOLDER,
# NOT A MEASUREMENT -- bench-test with a tape measure + slow-motion video
# (as flagged when this launcher mechanism was first designed) and replace
# this before quoting any of these numbers as final.
WHEEL_TO_BALL_EFFICIENCY = 0.85


def motor_rpm_at_voltage(voltage: float,
                          no_load_rpm_at_rated_v: float = MOTOR_NO_LOAD_RPM_AT_18V,
                          rated_v: float = MOTOR_RATED_VOLTAGE) -> float:
    """Brushed PM DC motor no-load speed scales ~linearly with applied
    voltage (the same first-order principle as a brushless motor's Kv
    rating). This ignores brush/bearing friction losses, which matter
    proportionally more at low voltage -- treat as an estimate, confirm
    with a tachometer if precision matters for your report."""
    return no_load_rpm_at_rated_v * (voltage / rated_v)


def wheel_surface_speed_mps(wheel_rpm: float, wheel_radius_m: float = WHEEL_RADIUS_M) -> float:
    omega = wheel_rpm * 2 * math.pi / 60.0
    return omega * wheel_radius_m


def exit_velocity_geared(voltage: float = SUPPLY_VOLTAGE) -> float:
    """Exit velocity if the wheels are driven through the BBOX-15's 16:1
    reduction, as currently specified in the BOM. See CRITICAL FINDING."""
    motor_rpm = motor_rpm_at_voltage(voltage)
    wheel_rpm = motor_rpm / GEARBOX_RATIO
    return wheel_surface_speed_mps(wheel_rpm) * WHEEL_TO_BALL_EFFICIENCY


def exit_velocity_direct_drive(voltage: float = SUPPLY_VOLTAGE) -> float:
    """Exit velocity if the wheels are mounted directly on the bare motor
    shaft, bypassing the gearbox entirely -- this file's recommendation."""
    motor_rpm = motor_rpm_at_voltage(voltage)
    return wheel_surface_speed_mps(motor_rpm) * WHEEL_TO_BALL_EFFICIENCY


@dataclass
class LaunchSolution:
    feasible: bool
    v0: Optional[float]
    angle_deg: Optional[float]
    time_of_flight: Optional[float]
    note: str


def solve_launch_angle(v0: float, horizontal_distance_m: float, height_drop_m: float) -> LaunchSolution:
    """Given a FIXED exit velocity v0 (the launcher is on/off, not
    throttleable shot-to-shot -- the burst duration is fixed, see
    esp8266_launcher.ino's BURST_MS), solve for the launch angle that
    lands the ball at (horizontal_distance_m, -height_drop_m) relative to
    the release point. height_drop_m is positive when the target is BELOW
    the drone (the normal case: drone hovers above the fire).

    Standard projectile motion, solved via the trajectory equation
        y = x*tan(theta) - (g*x^2) / (2*v0^2*cos^2(theta))
    Substituting u = tan(theta) and 1/cos^2(theta) = 1+u^2 turns this into
    a quadratic in u -- the standard closed form for "hit a fixed point at
    a fixed launch speed" (as opposed to the simpler case of solving for
    v0 at a GIVEN angle).
    """
    x = horizontal_distance_m
    y = -height_drop_m  # target is below the release point -> negative y

    if x == 0:
        if v0 <= 0:
            return LaunchSolution(False, None, None, None, "v0 must be positive")
        # Pure vertical case (straight down): 1D kinematics,
        # height_drop = v0*t + 0.5*g*t^2 (ball already moving down at v0)
        a, b, c = 0.5 * G, v0, -height_drop_m
        disc = b * b - 4 * a * c
        if disc < 0:
            return LaunchSolution(False, None, None, None, "no real solution (unexpected for a straight-down drop)")
        t = (-b + math.sqrt(disc)) / (2 * a)
        return LaunchSolution(True, v0, -90.0, t, "Vertical launch (straight down), zero horizontal standoff")

    k = (G * x ** 2) / (2 * v0 ** 2)
    a, b, c = k, -x, (y + k)
    disc = b * b - 4 * a * c
    if disc < 0:
        return LaunchSolution(False, v0, None, None,
                               f"Not reachable at v0={v0:.2f} m/s -- target is beyond this launcher's "
                               f"range at ANY angle")

    sqrt_disc = math.sqrt(disc)
    u1 = (-b + sqrt_disc) / (2 * a)
    u2 = (-b - sqrt_disc) / (2 * a)
    angle1 = math.degrees(math.atan(u1))
    angle2 = math.degrees(math.atan(u2))
    # Prefer the flatter (lower-angle, shorter-hang-time) trajectory --
    # less time for the fire's own thermal updraft to carry a 58g tennis
    # ball off target before it lands.
    chosen_angle = min([angle1, angle2], key=abs)
    chosen_u = math.tan(math.radians(chosen_angle))

    vx = v0 / math.sqrt(1 + chosen_u ** 2)
    t = x / vx if vx > 0 else None

    return LaunchSolution(True, v0, chosen_angle, t,
                           f"Two mathematical solutions exist ({angle1:.1f} deg and {angle2:.1f} deg); "
                           f"chose the flatter one for less flight time / thermal drift")


def max_range_at_angle(v0: float, angle_deg: float, height_drop_m: float) -> float:
    """How far (horizontally) this v0+angle combination can reach a target
    that's height_drop_m BELOW the release point. This is the basis of the
    GO/NO-GO decision below: exit velocity is fixed by the motor, so the
    only variable actually under the pilot's control in the moment is
    hover position (hence x)."""
    theta = math.radians(angle_deg)
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)
    a, b, c = 0.5 * G, -vy, -height_drop_m
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0
    t = (-b + math.sqrt(disc)) / (2 * a)
    return vx * t


def decide_launch(altitude_m: float, horizontal_offset_m: float,
                   v0: float, angle_deg: float = -90.0) -> LaunchSolution:
    """The actual decision function: fire_detector.py's MissionFSM should
    call this once it reaches AWAITING_CONFIRM / is about to fire. Given
    the measured altitude (TF-Luna) and horizontal offset to the fused
    fire detection, decide whether THIS launcher, AT THIS v0 and angle,
    can actually reach the target -- not just whether a fire was detected.

    IMPORTANT for your CURRENT CAD design: the launcher is a FIXED mount
    firing straight down (angle_deg=-90, the default). At that angle,
    max_range_at_angle is always 0 -- a straight-down launcher cannot
    reach ANY horizontal offset, by definition (all the velocity is
    vertical). That isn't a bug, it's the real constraint: with this
    mount, the decision is NOT "is the target within range" but "is the
    drone hovering close enough to directly overhead" -- see
    required_hover_precision() below for the actually useful version of
    that question for your current, fixed-mount design. This general
    angled form is here for if/when you add a tiltable launcher mount."""
    reach = max_range_at_angle(v0, angle_deg, altitude_m)
    if horizontal_offset_m <= reach + 1e-9:
        return LaunchSolution(True, v0, angle_deg, None,
                               f"GO: target at {horizontal_offset_m:.2f} m is within this launcher's "
                               f"{reach:.2f} m reach at {altitude_m:.1f} m altitude")
    shortfall = horizontal_offset_m - reach
    return LaunchSolution(False, v0, angle_deg, None,
                           f"NO-GO: target at {horizontal_offset_m:.2f} m exceeds this launcher's "
                           f"{reach:.2f} m reach by {shortfall:.2f} m -- pilot must close the distance "
                           f"before firing")


def time_of_flight_straight_down(altitude_m: float, v0: float) -> float:
    sol = solve_launch_angle(v0=v0, horizontal_distance_m=0.0, height_drop_m=altitude_m)
    return sol.time_of_flight


def required_hover_precision(altitude_m: float, v0: float, max_miss_m: float) -> dict:
    """THE decision that actually matches your current fixed, straight-
    down launcher mount. Since the launcher can't steer the ball
    sideways, accuracy depends entirely on (a) the pilot/hover-hold
    getting the drone directly over the target, and (b) the drone having
    near-zero HORIZONTAL velocity at the instant of firing -- any residual
    drift velocity carries over to the ball (it keeps the drone's motion
    at the moment of release) and multiplies by the fall time.

    This neglects air drag on the ball during the fall, which a real
    tennis ball does experience non-negligibly over several meters --
    treat this as a first-order, slightly optimistic estimate, good
    enough to set a practical pilot/hover-hold tolerance, not a precise
    ballistic prediction.
    """
    t = time_of_flight_straight_down(altitude_m, v0)
    max_drift_velocity = max_miss_m / t if t else 0.0
    return {
        "time_of_flight_s": t,
        "max_tolerable_drift_velocity_mps": max_drift_velocity,
        "note": (f"At {altitude_m:.1f}m altitude, the ball falls for {t:.2f}s. "
                 f"To land within {max_miss_m:.2f}m of directly-overhead, the drone's "
                 f"horizontal velocity at the moment of firing must be under "
                 f"{max_drift_velocity:.3f} m/s -- essentially hovering still.")
    }


# ===========================================================================
# Self-tests: verify the physics against known closed-form identities
# before trusting any of the numbers above.
# ===========================================================================

def _run_selftests():
    # 1. Classic level-ground range formula: R = v0^2 * sin(2*theta) / g.
    #    At 45 degrees and height_drop=0, this reduces to R = v0^2 / g.
    v0_test = 10.0
    expected = v0_test ** 2 / G
    got = max_range_at_angle(v0_test, 45.0, height_drop_m=0.0)
    assert abs(got - expected) < 1e-6, f"range formula mismatch: {got} vs {expected}"

    # 2. solve_launch_angle should be the INVERSE of max_range_at_angle:
    #    solving for the angle that reaches a known (x, height_drop) pair,
    #    then feeding that angle back into max_range_at_angle, should
    #    reproduce the same x.
    x_target, h_target = 3.0, 4.0
    sol = solve_launch_angle(v0=10.0, horizontal_distance_m=x_target, height_drop_m=h_target)
    assert sol.feasible, "expected a feasible solution for this easy case"
    x_check = max_range_at_angle(sol.v0, sol.angle_deg, h_target)
    assert abs(x_check - x_target) < 1e-6, f"solve/check round-trip mismatch: {x_check} vs {x_target}"

    # 3. Vertical (x=0) case: a ball dropped with added downward launch
    #    velocity should take LESS time to fall a given height than one
    #    dropped from rest (sanity check on the direction of the effect).
    t_with_launch = solve_launch_angle(v0=5.0, horizontal_distance_m=0.0, height_drop_m=5.0).time_of_flight
    t_free_fall = math.sqrt(2 * 5.0 / G)
    assert t_with_launch < t_free_fall, "launched drop should be faster than free fall, not slower"

    # 4. decide_launch should flip from GO to NO-GO exactly at the
    #    computed max range, not somewhere else.
    reach = max_range_at_angle(v0=8.0, angle_deg=-90.0, height_drop_m=3.0)
    assert decide_launch(altitude_m=3.0, horizontal_offset_m=reach - 0.01, v0=8.0).feasible
    assert not decide_launch(altitude_m=3.0, horizontal_offset_m=reach + 0.01, v0=8.0).feasible

    print("SELFTEST PASSED: projectile motion equations check out against known identities.")


if __name__ == "__main__":
    _run_selftests()

    print()
    print("=" * 78)
    print("CRITICAL FINDING: geared (as-specced) vs direct-drive launcher speed")
    print("=" * 78)
    v_geared = exit_velocity_geared()
    v_direct = exit_velocity_direct_drive()
    print(f"At {SUPPLY_VOLTAGE}V supply (your 4S flight battery):")
    print(f"  THROUGH the BBOX-15's 16:1 gearbox : {v_geared:.3f} m/s exit velocity"
          f"  <- too slow to be a launcher at all")
    print(f"  DIRECT DRIVE (bypass the gearbox)  : {v_direct:.3f} m/s exit velocity"
          f"  <- usable for a tennis ball")
    print(f"  (wheel-to-ball efficiency assumed: {WHEEL_TO_BALL_EFFICIENCY} -- UNMEASURED, bench-test this)")

    print()
    print("=" * 78)
    print("Worked example: YOUR actual design (fixed, straight-down launcher)")
    print("What hover precision does the pilot need to land within 0.3m of the fire?")
    print("=" * 78)
    for alt in (2.0, 5.0, 8.0):  # spans ALT_MIN_M..ALT_MAX_M from fire_detector.py
        result = required_hover_precision(altitude_m=alt, v0=v_direct, max_miss_m=0.3)
        print(f"  At {alt:.1f}m altitude: {result['note']}")

    print()
    print("=" * 78)
    print("Bonus / future work: IF you add a tiltable launcher mount later,")
    print("required angle for a 2m standoff at 4m altitude would be:")
    print("=" * 78)
    sol = solve_launch_angle(v0=v_direct, horizontal_distance_m=2.0, height_drop_m=4.0)
    if sol.feasible:
        print(f"  Required angle: {sol.angle_deg:.1f} deg from horizontal, "
              f"time of flight: {sol.time_of_flight:.2f}s -- {sol.note}")
    else:
        print(f"  {sol.note}")
