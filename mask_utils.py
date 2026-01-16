"""
mask_utils.py

Utility functions for applying spatial masks to detection images.

Currently supports:
- Circular region-of-interest (ROI) masking controlled by config.ENABLE_MASK

This module is intentionally small and isolated so masking logic
does not leak into detector or tracking code.
"""

import numpy as np
import config


def apply_circular_mask(mask):
    """
    Apply a circular ROI mask to a binary foreground mask.

    Parameters
    ----------
    mask : np.ndarray (uint8)
        Binary foreground mask (0 or 255).

    Returns
    -------
    masked : np.ndarray (uint8)
        Mask with pixels outside the circular ROI set to zero.
        If ENABLE_MASK is False, the input mask is returned unchanged.
    """
    if not config.ENABLE_MASK:
        return mask

    h, w = mask.shape

    # Create coordinate grids
    Y, X = np.ogrid[:h, :w]

    # Squared distance from mask center
    dist2 = (X - config.MASK_XC) ** 2 + (Y - config.MASK_YC) ** 2

    # Circular ROI
    circle = dist2 <= (config.MASK_RADIUS ** 2)

    # Apply ROI (preserve dtype)
    return mask * circle.astype(mask.dtype)
