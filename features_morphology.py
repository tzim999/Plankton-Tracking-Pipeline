"""
features_morphology.py

Morphology-derived feature extraction from segmented contours.

This module computes stable, low-noise shape descriptors that
carry strong biological meaning and are suitable for behavior
state classification.

All measurements are per-frame and per-object.
"""

import cv2
import numpy as np


def morphology_features(contour):
    """
    Extract core morphology features from a contour.

    Parameters
    ----------
    contour : np.ndarray
        OpenCV contour for a detected organism.

    Returns
    -------
    features : dict
        Dictionary containing morphology-derived features.
    """
    # Area and perimeter
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)

    # Minimum-area bounding box
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), angle = rect

    # Aspect ratio (long / short axis)
    long_axis = max(w, h)
    short_axis = min(w, h) + 1e-6
    aspect_ratio = long_axis / short_axis

    # Convex hull solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / hull_area if hull_area > 0 else 0.0

    # Body axis angle (OpenCV convention)
    body_axis_angle_deg = angle

    return {
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "body_axis_angle_deg": body_axis_angle_deg,
    }


