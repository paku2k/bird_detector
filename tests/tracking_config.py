#!/usr/bin/env python3
"""
Quick configuration helper for test_advanced_tracking.py

Edit the settings at the top of this file, then the test script
will read these values.
"""

# ==================== TRACKING CONFIGURATION ====================

# Tracker type: "MOSSE" (fastest) or "CSRT" (most accurate)
TRACKER_TYPE = "MOSSE"

# History tracking mode:
# 1 = Track through history frames from detection point to present
# 0 = Initialize tracker on current frame only
HISTORY_TRACK = 1

# YOLO validation interval in seconds
# How often should YOLO re-check if tracker is still on target?
YOLO_INTERVAL = 2.0

# IOU threshold for tracker validation
# If IOU between YOLO detection and tracker box drops below this, reinit tracker
IOU_THRESHOLD = 0.3

# Frame history buffer size (number of frames to keep)
FRAME_HISTORY_SIZE = 60

# Bbox shrink percentage (0.0 to 0.5)
# Shrinks bbox by this % on each side to focus on bird body, not background
SHRINK_BBOX_PERCENT = 0.15

# ==================== VISUALIZATION ====================

SHOW_DETECTION_BOX = True
SHOW_TRACKER_BOX = True
SHOW_FEATURES = True

# Flask web server port
FLASK_PORT = 5000

# ==================== PRESETS ====================

def preset_fast():
    """Fast tracking preset (MOSSE, no history)"""
    global TRACKER_TYPE, HISTORY_TRACK, YOLO_INTERVAL
    TRACKER_TYPE = "MOSSE"
    HISTORY_TRACK = 0
    YOLO_INTERVAL = 3.0

def preset_accurate():
    """Accurate tracking preset (CSRT, with history)"""
    global TRACKER_TYPE, HISTORY_TRACK, YOLO_INTERVAL
    TRACKER_TYPE = "CSRT"
    HISTORY_TRACK = 1
    YOLO_INTERVAL = 2.0

def preset_paranoid():
    """Paranoid preset (frequent YOLO checks, high IOU threshold)"""
    global TRACKER_TYPE, YOLO_INTERVAL, IOU_THRESHOLD
    TRACKER_TYPE = "CSRT"
    YOLO_INTERVAL = 1.0
    IOU_THRESHOLD = 0.5

# Uncomment one to use a preset:
# preset_fast()
# preset_accurate()
# preset_paranoid()

if __name__ == "__main__":
    print("Current Configuration:")
    print(f"  TRACKER_TYPE: {TRACKER_TYPE}")
    print(f"  HISTORY_TRACK: {HISTORY_TRACK}")
    print(f"  YOLO_INTERVAL: {YOLO_INTERVAL}s")
    print(f"  IOU_THRESHOLD: {IOU_THRESHOLD}")
    print(f"  FRAME_HISTORY_SIZE: {FRAME_HISTORY_SIZE}")
    print(f"  SHRINK_BBOX_PERCENT: {SHRINK_BBOX_PERCENT}")
