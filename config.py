# Copyright (c) 2025 Thomas Zimmerman — MIT License
"""
config.py
Central configuration file for the plankton tracking pipeline.
All tuning parameters should be modified here rather than in code.
"""

# Video / timing
FPS = 30
PLAYBACK_DELAY = 1   # milliseconds; 1 = real-time-ish, 0 = pause
HREZ = 640  # Display resolution
VREZ = 512

# Background modeling
BG_SAMPLES = 100   # number of frames used to build median background

# Circular detection mask
ENABLE_MASK = False      # True = apply circular ROI, False = full frame
MASK_XC = 320
MASK_YC = 256
MASK_RADIUS = 128

# Detection / thresholding
THRESH_METHOD = "fixed"   # "fixed" or "otsu"
THRESH = 10
MIN_A = 10
MAX_A = 824
MIN_WH = 2
MAX_WH = 52
MIN_FRAME_DISPLACEMENT_PX = 0.5
DETECT_WINDOW_FRAMES = 5
DETECT_MIN_COUNT = 3


# Tracking
MAX_HISTORY = 50        # stored centroid history (memory budget)
MAX_TRACK_DIST = 50
MAX_LOST_FRAMES = 10

# Motion / behavior
VELOCITY_WINDOW_FRAMES = 15    # window for averaged velocity
K_CONSECUTIVE_FRAMES = 3      # anti-flicker requirement
MIN_MOVEMENT_SPEED_PX_S = 8.0 # estimated from jitter + FPS
BALL_ASPECT_EPS = 0.25
NUM_CAP_LINES = 4
MIN_DISPLACEMENT_PX = 2.0   # net displacement gate for windowed velocity

# Notes:
# - Velocity ramps smoothly over VELOCITY_WINDOW_FRAMES
# - Movement states: SWIM_FORWARD, SWIM_BACKWARD, ATTACHED
# - No IMMOBILE state (removed by background subtraction)
