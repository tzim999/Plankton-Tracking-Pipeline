"""
movement_state.py

Movement state classification with temporal stability.

Defines three mutually exclusive movement states:
    - SWIM_FORWARD
    - SWIM_BACKWARD
    - ATTACHED

Key design features:
- Uses window-averaged speed (no instantaneous velocity)
- Requires K consecutive frames above threshold to enter swimming
- BALL pose is allowed to swim but has no polarity
- No IMMOBILE state (removed by background subtraction)
"""

import config


class MovementStateTracker:
    """
    Tracks per-object movement state using K-consecutive-frame logic.
    """

    def __init__(self):
        # Counts how many consecutive frames an object exceeds speed threshold
        self._above_thresh_count = {}

    def update(self, object_id, speed_avg_px_s, pose, head_aligned_forward):
        """
        Update and return movement state for a tracked object.

        Parameters
        ----------
        object_id : int
            Persistent ID for the object.
        speed_avg_px_s : float or None
            Window-averaged speed in pixels/sec.
        pose : str
            "ELONGATED" or "BALL".
        head_aligned_forward : bool
            True if head direction aligns with motion direction (≤ 90°).

        Returns
        -------
        movement_state : str or None
            One of: "SWIM_FORWARD", "SWIM_BACKWARD", "ATTACHED".
            Returns None if speed is unavailable (insufficient history).
        """
        # Not enough history yet
        if speed_avg_px_s is None:
            return None

        # Initialize counter if needed
        if object_id not in self._above_thresh_count:
            self._above_thresh_count[object_id] = 0

        # Update consecutive-frame counter
        if speed_avg_px_s >= config.MIN_MOVEMENT_SPEED_PX_S:
            self._above_thresh_count[object_id] += 1
        else:
            self._above_thresh_count[object_id] = 0

        # Not yet swimming (anti-flicker)
        if self._above_thresh_count[object_id] < config.K_CONSECUTIVE_FRAMES:
            return "ATTACHED"

        # Swimming state reached
        if pose == "BALL":
            # No polarity, but still real swimming
            return "SWIM_FORWARD"

        # Elongated: forward or backward
        if head_aligned_forward:
            return "SWIM_FORWARD"
        else:
            return "SWIM_BACKWARD"
