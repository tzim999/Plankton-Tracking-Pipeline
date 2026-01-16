"""
features_motion.py

Motion-derived feature extraction.

This module separates:
1) Binary motion detection (robust, conservative)
2) Continuous velocity measurement (smooth, causal)

Key properties:
- No instantaneous velocity
- Binary motion gate suppresses jitter
- Velocity ramps smoothly once motion is detected
"""

import numpy as np
import config


def binary_motion_detect(recent_deltas):
    """
    Decide whether an object is moving based on recent
    per-frame centroid displacements.

    Parameters
    ----------
    recent_deltas : list of float
        Per-frame centroid displacements (pixels).

    Returns
    -------
    is_moving : bool
        True if motion is detected, False otherwise.
    """
    if recent_deltas is None:
        return False

    if len(recent_deltas) < config.DETECT_WINDOW_FRAMES:
        return False

    count = sum(
        d >= config.MIN_FRAME_DISPLACEMENT_PX
        for d in recent_deltas
    )

    return count >= config.DETECT_MIN_COUNT


def windowed_velocity(history):
    """
    Compute window-averaged translational speed and direction.

    Parameters
    ----------
    history : list of (x, y)
        Centroid history for a tracked object, ordered in time.

    Returns
    -------
    speed_avg_px_s : float or None
        Window-averaged speed in pixels per second.
    velocity_angle_deg : float or None
        Direction of net displacement in degrees.
    """
    N = config.VELOCITY_WINDOW_FRAMES

    # Require N+1 samples to compute displacement
    if history is None or len(history) < (N + 1):
        return None, None

    # Net displacement over window
    x0, y0 = history[-(N + 1)]
    x1, y1 = history[-1]

    dx = x1 - x0
    dy = y1 - y0

    distance_px = np.hypot(dx, dy)

    # Suppress sub-pixel / jitter motion explicitly
    if distance_px < config.MIN_DISPLACEMENT_PX:
        return 0.0, None

    speed_avg_px_s = distance_px * config.FPS / N
    velocity_angle_deg = np.degrees(np.arctan2(dy, dx))

    return speed_avg_px_s, velocity_angle_deg
