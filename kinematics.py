# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
kinematics.py

Morphology-based pose and polarity inference.

This module determines:
- Pose classification (BALL vs ELONGATED)
- Head vs tail polarity (for elongated organisms only)
- Body-axis orientation (tail → head)

Important design constraints:
- No velocity or motion computation (handled elsewhere)
- No biological state inference (handled elsewhere)
- BALL pose explicitly has no polarity
"""

import cv2
import numpy as np
import config


# ------------------------------------------------------------
# Pose classification
# ------------------------------------------------------------
def classify_pose(long_axis, short_axis):
    """
    Classify pose based on aspect ratio.

    Parameters
    ----------
    long_axis : float
    short_axis : float

    Returns
    -------
    pose : str
        "BALL" or "ELONGATED"
    """
    ratio = long_axis / (short_axis + 1e-6)

    if abs(ratio - 1.0) < config.BALL_ASPECT_EPS:
        return "BALL"
    else:
        return "ELONGATED"


# ------------------------------------------------------------
# Head / tail inference from bounding box
# ------------------------------------------------------------
def head_tail_from_bbox(mask, contour):
    """
    Determine pose, head, tail, and body-axis direction
    using rotated bounding-box cap integration.

    Parameters
    ----------
    mask : np.ndarray (uint8)
        Binary foreground mask.
    contour : np.ndarray
        Contour of the detected organism.

    Returns
    -------
    pose : str
        "BALL" or "ELONGATED"
    head : (int, int) or None
        Head location in image coordinates.
    tail : (int, int) or None
        Tail location in image coordinates.
    axis_vec : np.ndarray shape (2,)
        Unit vector along body axis (tail → head).
        Meaningful even for BALL pose (orientation only).
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # Ensure long_axis >= short_axis
    if w >= h:
        long_axis = w
        short_axis = h
        theta = np.deg2rad(angle)
    else:
        long_axis = h
        short_axis = w
        theta = np.deg2rad(angle + 90)

    # Body-axis unit vectors
    u = np.array([np.cos(theta), np.sin(theta)])   # long axis
    v = np.array([-np.sin(theta), np.cos(theta)])  # short axis

    # Pose classification
    pose = classify_pose(long_axis, short_axis)

    # BALL: no polarity
    if pose == "BALL":
        return pose, None, None, u

    # --------------------------------------------------------
    # Identify short edges ("caps") of the bounding box
    # --------------------------------------------------------
    box = cv2.boxPoints(rect).astype(np.float32)

    edges = [
        (box[i], box[(i + 1) % 4])
        for i in range(4)
    ]

    lengths = [
        np.linalg.norm(p1 - p0)
        for p0, p1 in edges
    ]

    short_edges = [edges[i] for i in np.argsort(lengths)[:2]]

    # Midpoints of the short edges
    cap_midpoints = [
        (p0 + p1) / 2.0
        for (p0, p1) in short_edges
    ]

    # --------------------------------------------------------
    # Multi-line cap integration
    # --------------------------------------------------------
    def integrate_cap(cap_mid, direction):
        """
        Integrate foreground pixels inward from a cap
        across multiple width-parallel lines.
        """
        total = 0

        for k in range(config.NUM_CAP_LINES):
            center = cap_mid + direction * k

            for t in np.linspace(-short_axis / 2,
                                 short_axis / 2,
                                 int(short_axis) + 1):
                pt = center + t * v
                x = int(round(pt[0]))
                y = int(round(pt[1]))

                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    if mask[y, x] > 0:
                        total += 1

        return total

    # Integrate inward from each cap
    sum_a = integrate_cap(cap_midpoints[0], +u)
    sum_b = integrate_cap(cap_midpoints[1], -u)

    # --------------------------------------------------------
    # Head vs tail decision
    # --------------------------------------------------------
    if sum_a >= sum_b:
        head_cap = cap_midpoints[0]
        tail_cap = cap_midpoints[1]
        axis_vec = u
    else:
        head_cap = cap_midpoints[1]
        tail_cap = cap_midpoints[0]
        axis_vec = -u

    head = (int(round(head_cap[0])), int(round(head_cap[1])))
    tail = (int(round(tail_cap[0])), int(round(tail_cap[1])))

    return pose, head, tail, axis_vec
