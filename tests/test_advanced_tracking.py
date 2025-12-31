#!/usr/bin/env python3
"""
test_advanced_tracking.py

Advanced bird tracking test with history-based tracker initialization,
background YOLO validation, and Flask web visualization.

Features:
1. YOLO detector continuously scans for birds
2. When bird found, activates tracker (MOSSE/CSRT)
3. Two tracker initialization modes:
   - HISTORY_TRACK=1: Track through history frames from detection point to present
   - HISTORY_TRACK=0: Initialize on current frame only
4. Background YOLO validation with IOU checking
5. Live Flask web visualization
"""

import sys
import os
import cv2
import time
import threading
import queue
from collections import deque
from flask import Flask, Response
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.yolo import YoloDetector
from src.tracking import ObjectTracker
import config

# Create tmp directory for debug images
TMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# ==================== CONFIGURATION ====================
TRACKER_TYPE = "CSRT"
HISTORY_TRACK = 0
YOLO_INTERVAL = 2
IOU_THRESHOLD = 0.4             # Minimum IOU to consider tracker valid
FRAME_HISTORY_SIZE = 60         # Number of frames to keep in history buffer
SHRINK_BBOX_PERCENT = 0.10      # Shrink bbox by this % to focus on bird body
MAX_TIME_NO_DETECTION = 10.0    # Seconds to keep tracking even if YOLO loses bird

# Visualization settings
SHOW_DETECTION_BOX = True
SHOW_TRACKER_BOX = True
SHOW_FEATURES = True

# ==================== GLOBAL STATE ====================
app = Flask(__name__)

class TrackingState:
    def __init__(self):
        self.mode = "SEARCHING"  # SEARCHING, TRACKING, LOST
        self.current_frame = None
        self.display_frame = None
        self.frame_lock = threading.Lock()
        
        # Frame history with timestamps and IDs
        self.frame_history = deque(maxlen=FRAME_HISTORY_SIZE)
        self.frame_counter = 0
        
        # Detection info
        self.last_detection_box = None
        self.last_detection_score = 0.0
        self.last_detection_frame_id = None
        
        # Tracker info
        self.tracker_box = None
        self.tracker_active = False
        
        # YOLO validation
        self.last_yolo_check_time = 0
        self.yolo_queue = queue.Queue()
        self.yolo_busy = False
        
        # Statistics
        self.stats = {
            "detections": 0,
            "tracks": 0,
            "lost_tracks": 0,
            "reinitialized": 0,
            "avg_tracker_fps": 0.0,
            "avg_yolo_fps": 0.0,
            "avg_yolo_ms": 0.0
        }
        
        # Timing for no-detection tolerance
        self.last_successful_detection_time = 0
        
        # Debug tracking trajectory
        self.current_trajectory_dir = None
        self.trajectory_frame_count = 0

state = TrackingState()

# ==================== FRAME HISTORY MANAGEMENT ====================
def add_frame_to_history(frame, frame_id):
    """Add frame to history buffer with metadata"""
    with state.frame_lock:
        state.frame_history.append({
            'frame': frame.copy(),
            'frame_id': frame_id,
            'timestamp': time.time()
        })

def get_frame_by_id(frame_id):
    """Retrieve a specific frame from history"""
    with state.frame_lock:
        for entry in state.frame_history:
            if entry['frame_id'] == frame_id:
                return entry['frame']
    return None

def get_frames_since(frame_id):
    """Get all frames from frame_id to present"""
    with state.frame_lock:
        frames = []
        found_start = False
        for entry in state.frame_history:
            if entry['frame_id'] == frame_id:
                found_start = True
            if found_start:
                frames.append(entry)
        return frames

# ==================== BOUNDING BOX UTILITIES ====================
def shrink_bbox(bbox, shrink_percent, frame_shape):
    """Shrink bbox to focus on center (bird body only)"""
    x, y, w, h = bbox
    h_img, w_img = frame_shape[:2]
    
    shrink_w = int(w * shrink_percent)
    shrink_h = int(h * shrink_percent)
    
    x_new = x + shrink_w
    y_new = y + shrink_h
    w_new = w - (shrink_w * 2)
    h_new = h - (shrink_h * 2)
    
    # Boundary checks
    x_new = max(0, x_new)
    y_new = max(0, y_new)
    w_new = max(5, min(w_new, w_img - x_new))
    h_new = max(5, min(h_new, h_img - y_new))
    
    return (x_new, y_new, w_new, h_new)

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

# ==================== YOLO BACKGROUND WORKER ====================
def yolo_worker(detector):
    """Background thread for YOLO validation"""
    print("🧠 YOLO validation worker started")
    
    while True:
        try:
            # Wait for validation request
            task = state.yolo_queue.get(timeout=0.5)
            
            if task is None:  # Shutdown signal
                break
            
            frame, frame_id = task
            state.yolo_busy = True
            
            yolo_start = time.time()
            found, score, box = detector.detect(frame)
            yolo_end = time.time()
            yolo_time = (yolo_end - yolo_start) * 1000
            
            # Update stats
            state.stats['avg_yolo_ms'] = yolo_time
            state.stats['avg_yolo_fps'] = 1000.0 / yolo_time if yolo_time > 0 else 0
            
            if found:
                print(f"  ✓ YOLO validation: Bird detected (score={score:.2f}, time={yolo_time:.0f}ms, {state.stats['avg_yolo_fps']:.1f} FPS)")
                
                # Update last successful detection time
                state.last_successful_detection_time = time.time()
                
                # Check IOU with current tracker position
                if state.tracker_box is not None:
                    iou = calculate_iou(state.tracker_box, box)
                    print(f"    IOU with tracker: {iou:.3f} (threshold={IOU_THRESHOLD})")
                    
                    if iou < IOU_THRESHOLD:
                        print(f"    ⚠️  Tracker drift detected! IOU too low. Flagging for reinitialization.")
                        state.last_detection_box = box
                        state.last_detection_score = score
                        state.last_detection_frame_id = frame_id
                        state.mode = "REINIT_NEEDED"
                    else:
                        print(f"    ✓ Tracker still valid")
                else:
                    # Tracker was lost, we have a new detection
                    state.last_detection_box = box
                    state.last_detection_score = score
                    state.last_detection_frame_id = frame_id
            else:
                print(f"  ✗ YOLO validation: No bird found (time={yolo_time:.0f}ms, {state.stats['avg_yolo_fps']:.1f} FPS)")
                
                if state.mode == "TRACKING":
                    # Check if we've been without detection for too long
                    time_since_detection = time.time() - state.last_successful_detection_time
                    print(f"    Time since last detection: {time_since_detection:.1f}s (max: {MAX_TIME_NO_DETECTION}s)")
                    
                    if time_since_detection > MAX_TIME_NO_DETECTION:
                        print(f"    ⚠️  No detection for {time_since_detection:.1f}s. Aborting tracking.")
                        state.mode = "LOST"
                    else:
                        print(f"    ℹ️  Continuing to track (grace period active)")
            
            state.yolo_busy = False
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ YOLO worker error: {e}")
            state.yolo_busy = False

# ==================== DEBUG IMAGE SAVING ====================
def save_debug_image(frame, bbox, label, save_dir):
    return
    """Save debug image with bounding box"""
    debug_frame = frame.copy()
    
    # Convert grayscale to BGR if needed
    if len(debug_frame.shape) == 2:
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding box
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add label
    cv2.putText(debug_frame, label, (x, y-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add bbox info
    info_text = f"Box: ({x},{y},{w},{h})"
    cv2.putText(debug_frame, info_text, (x, y+h+20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add frame info
    frame_text = f"Frame: {state.frame_counter}"
    cv2.putText(debug_frame, frame_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Save to specified directory
    filename = f"{label}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, debug_frame)
    print(f"   📸 Saved: {filename}")
    return filepath

def create_trajectory_directory():
    """Create a new directory for the current tracking trajectory"""
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trajectory_dir = os.path.join(TMP_DIR, f"trajectory_{unique_id}")
    os.makedirs(trajectory_dir, exist_ok=True)
    print(f"   📁 Created trajectory folder: {os.path.basename(trajectory_dir)}")
    return trajectory_dir

# ==================== TRACKER INITIALIZATION ====================
def initialize_tracker_with_history(tracker, detection_frame_id, bbox):
    """
    Initialize tracker and track through history frames (HISTORY_TRACK=1)
    """
    print(f"\n🔄 Initializing tracker with HISTORY tracking...")
    print(f"   Detection was at frame_id={detection_frame_id}")
    print(f"   Current frame_id={state.frame_counter}")
    
    # Create trajectory directory for this tracking session
    state.current_trajectory_dir = create_trajectory_directory()
    state.trajectory_frame_count = 0
    
    # Get the detection frame
    detection_frame = get_frame_by_id(detection_frame_id)
    if detection_frame is None:
        print(f"   ⚠️  Detection frame not in history! Using current frame.")
        current_frame = state.current_frame.copy()
        tracker.start(current_frame, bbox)
        return True
    
    # Shrink bbox to focus on bird body
    bbox_shrunk = shrink_bbox(bbox, SHRINK_BBOX_PERCENT, detection_frame.shape)
    print(f"   Original bbox: {bbox}")
    print(f"   Shrunk bbox (focus on body): {bbox_shrunk}")
    
    # SAVE START IMAGE
    save_debug_image(detection_frame, bbox_shrunk, 
                    f"000_start_history_frame{detection_frame_id}", 
                    state.current_trajectory_dir)
    
    # Initialize tracker on detection frame
    tracker.start(detection_frame, bbox_shrunk)
    
    # Get all frames from detection to present
    history_frames = get_frames_since(detection_frame_id)
    
    if len(history_frames) <= 1:
        print(f"   ℹ️  No history to track through (already at current frame)")
        return True
    
    print(f"   Tracking through {len(history_frames)-1} history frames...")
    
    # Track through history
    tracked_count = 0
    for i, entry in enumerate(history_frames[1:], 1):
        frame = entry['frame']
        frame_id = entry['frame_id']
        success, box = tracker.update(frame)
        
        if not success:
            print(f"   ⚠️  Tracker lost at history frame {i}/{len(history_frames)-1}")
            return False
        
        tracked_count += 1
        
        # Save every 5th frame during history tracking
        if i % 1 == 0:
            save_debug_image(frame, box, 
                           f"{i:03d}_history_frame{frame_id}",
                           state.current_trajectory_dir)
    
    # SAVE END OF HISTORY IMAGE
    final_entry = history_frames[-1]
    final_frame = final_entry['frame']
    success, final_box = tracker.update(final_frame)
    if success:
        save_debug_image(final_frame, final_box, 
                        f"{tracked_count:03d}_end_history_frame{state.frame_counter}",
                        state.current_trajectory_dir)
    
    print(f"   ✓ Successfully tracked through {tracked_count} history frames!")
    print(f"   Tracker is now at current frame")
    
    return True

def initialize_tracker_current_frame(tracker, bbox):
    """
    Initialize tracker on current frame only (HISTORY_TRACK=0)
    """
    print(f"\n🎯 Initializing tracker on CURRENT frame...")
    
    # Create trajectory directory for this tracking session
    state.current_trajectory_dir = create_trajectory_directory()
    state.trajectory_frame_count = 0
    
    current_frame = state.current_frame.copy()
    
    # Shrink bbox to focus on bird body
    bbox_shrunk = shrink_bbox(bbox, SHRINK_BBOX_PERCENT, current_frame.shape)
    print(f"   Original bbox: {bbox}")
    print(f"   Shrunk bbox (focus on body): {bbox_shrunk}")
    
    # SAVE START IMAGE
    save_debug_image(current_frame, bbox_shrunk, 
                    f"000_start_frame{state.frame_counter}",
                    state.current_trajectory_dir)
    
    tracker.start(current_frame, bbox_shrunk)
    print(f"   ✓ Tracker initialized")
    
    return True

# ==================== VISUALIZATION ====================
def create_visualization_frame(frame, mode, tracker_box, detection_box, detection_score):
    """Create annotated frame for web display"""
    vis_frame = frame.copy()
    
    # Convert grayscale to BGR if needed
    if len(vis_frame.shape) == 2:
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
    
    h, w = vis_frame.shape[:2]
    
    # Draw detection box (red)
    if SHOW_DETECTION_BOX and detection_box is not None:
        x, y, bw, bh = detection_box
        cv2.rectangle(vis_frame, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        cv2.putText(vis_frame, f"YOLO: {detection_score:.2f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw tracker box (green)
    if SHOW_TRACKER_BOX and tracker_box is not None:
        x, y, bw, bh = tracker_box
        cv2.rectangle(vis_frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(vis_frame, "TRACKER", 
                   (x, y+bh+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw center crosshair
        cx, cy = x + bw//2, y + bh//2
        cv2.circle(vis_frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.line(vis_frame, (cx-10, cy), (cx+10, cy), (0, 255, 0), 2)
        cv2.line(vis_frame, (cx, cy-10), (cx, cy+10), (0, 255, 0), 2)
    
    # Status overlay
    status_color = {
        "SEARCHING": (255, 165, 0),   # Orange
        "TRACKING": (0, 255, 0),       # Green
        "LOST": (0, 0, 255),           # Red
        "REINIT_NEEDED": (255, 255, 0) # Yellow
    }.get(mode, (255, 255, 255))
    
    cv2.rectangle(vis_frame, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(vis_frame, f"MODE: {mode}", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(vis_frame, f"Tracker: {TRACKER_TYPE} ({state.stats['avg_tracker_fps']:.1f} FPS) | History: {HISTORY_TRACK}", 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis_frame, f"YOLO: {state.stats['avg_yolo_fps']:.1f} FPS ({state.stats['avg_yolo_ms']:.0f}ms) | Frame: {state.frame_counter}", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show time since last detection when tracking
    if mode == "TRACKING" and state.last_successful_detection_time > 0:
        time_since = time.time() - state.last_successful_detection_time
        grace_color = (0, 255, 0) if time_since < MAX_TIME_NO_DETECTION else (0, 0, 255)
        cv2.putText(vis_frame, f"Time since YOLO: {time_since:.1f}s / {MAX_TIME_NO_DETECTION}s", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, grace_color, 1)
    
    # Stats overlay
    stats_y = h - 60
    cv2.rectangle(vis_frame, (0, stats_y), (w, h), (0, 0, 0), -1)
    cv2.putText(vis_frame, f"Detections: {state.stats['detections']} | Tracks: {state.stats['tracks']} | Lost: {state.stats['lost_tracks']} | Reinit: {state.stats['reinitialized']}", 
               (10, stats_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis_frame, f"IOU threshold: {IOU_THRESHOLD} | YOLO interval: {YOLO_INTERVAL}s | Grace period: {MAX_TIME_NO_DETECTION}s", 
               (10, stats_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_frame

def generate_frames():
    """Flask video stream generator"""
    while True:
        if state.display_frame is not None:
            ret, buffer = cv2.imencode('.jpg', state.display_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30 FPS max for web stream

@app.route('/')
def index():
    return """
    <html>
    <head><title>Advanced Bird Tracking Test</title></head>
    <body style="background: #1a1a1a; color: white; font-family: monospace;">
        <h1>🎯 Advanced Bird Tracking Test</h1>
        <img src="/video_feed" style="width: 100%; max-width: 1280px; border: 2px solid #00ff00;">
        <div style="margin-top: 20px;">
            <h3>Controls:</h3>
            <p>Tracker: """ + TRACKER_TYPE + """<br>
            History Mode: """ + str(HISTORY_TRACK) + """ (1=History tracking, 0=Current frame)<br>
            YOLO Interval: """ + str(YOLO_INTERVAL) + """s<br>
            IOU Threshold: """ + str(IOU_THRESHOLD) + """<br>
            Grace Period: """ + str(MAX_TIME_NO_DETECTION) + """s (keeps tracking even if YOLO loses bird)</p>
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== MAIN TRACKING LOOP ====================
def main_tracking_loop(camera, detector):
    """Main tracking state machine"""
    print("\n" + "="*60)
    print("🎯 STARTING ADVANCED TRACKING TEST")
    print("="*60)
    print(f"Tracker Type: {TRACKER_TYPE}")
    print(f"History Tracking: {'ENABLED' if HISTORY_TRACK else 'DISABLED'}")
    print(f"YOLO Validation Interval: {YOLO_INTERVAL}s")
    print(f"IOU Threshold: {IOU_THRESHOLD}")
    print(f"Frame History Buffer: {FRAME_HISTORY_SIZE} frames")
    print(f"Max Time Without Detection: {MAX_TIME_NO_DETECTION}s")
    print("="*60 + "\n")
    
    tracker = ObjectTracker(tracker_type=TRACKER_TYPE)
    
    fps_tracker = deque(maxlen=30)
    last_fps_print = time.time()
    
    while True:
        loop_start = time.time()
        
        # Get current frame
        frame = camera.read() if config.USE_GRAYSCALE else camera.read_original()
        if frame is None:
            time.sleep(0.01)
            continue
        
        state.current_frame = frame.copy()
        state.frame_counter += 1
        
        # Add to history buffer
        add_frame_to_history(frame, state.frame_counter)
        
        # ===== STATE MACHINE =====
        
        if state.mode == "SEARCHING":
            # Continuously scan for birds with YOLO
            found, score, box = detector.detect(frame)
            
            if found:
                print(f"\n{'='*60}")
                print(f"🐦 BIRD DETECTED!")
                print(f"   Frame ID: {state.frame_counter}")
                print(f"   Confidence: {score:.3f}")
                print(f"   Box: {box}")
                print(f"{'='*60}")
                
                state.stats['detections'] += 1
                state.last_detection_box = box
                state.last_detection_score = score
                state.last_detection_frame_id = state.frame_counter
                state.last_successful_detection_time = time.time()
                
                # Initialize tracker based on HISTORY_TRACK setting
                if HISTORY_TRACK:
                    success = initialize_tracker_with_history(tracker, state.frame_counter, box)
                else:
                    success = initialize_tracker_current_frame(tracker, box)
                
                if success:
                    state.mode = "TRACKING"
                    state.tracker_active = True
                    state.stats['tracks'] += 1
                    state.last_yolo_check_time = time.time()
                    print(f"\n✓ Tracker activated!")
                else:
                    print(f"\n✗ Failed to initialize tracker")
            
            # Visualization for searching mode
            vis_frame = create_visualization_frame(frame, state.mode, None, None, 0.0)
            state.display_frame = vis_frame
            
        elif state.mode == "TRACKING":
            # Update tracker
            success, box = tracker.update(frame)
            
            if success:
                state.tracker_box = box
                state.trajectory_frame_count += 1
                
                # Save every 5th frame during live tracking
                if state.trajectory_frame_count % 1 == 0 and state.current_trajectory_dir:
                    save_debug_image(frame, box,
                                   f"{state.trajectory_frame_count:03d}_tracking_frame{state.frame_counter}",
                                   state.current_trajectory_dir)
                
                # Check if it's time for YOLO validation
                current_time = time.time()
                if (current_time - state.last_yolo_check_time >= YOLO_INTERVAL and 
                    not state.yolo_busy):
                    
                    print(f"\n⏰ Time for YOLO validation (last check {current_time - state.last_yolo_check_time:.1f}s ago)")
                    state.yolo_queue.put((frame.copy(), state.frame_counter))
                    state.last_yolo_check_time = current_time
                
                # Visualization
                vis_frame = create_visualization_frame(frame, state.mode, box, 
                                                      state.last_detection_box, 
                                                      state.last_detection_score)
                state.display_frame = vis_frame
                
            else:
                print(f"\n⚠️  TRACKER LOST!")
                
                # Save final frame where tracker was lost
                if state.current_trajectory_dir and state.tracker_box:
                    save_debug_image(frame, state.tracker_box,
                                   f"{state.trajectory_frame_count:03d}_LOST_frame{state.frame_counter}",
                                   state.current_trajectory_dir)
                
                state.mode = "LOST"
                state.tracker_active = False
                state.tracker_box = None
                state.stats['lost_tracks'] += 1
                state.current_trajectory_dir = None
        
        elif state.mode == "REINIT_NEEDED":
            print(f"\n🔄 Reinitializing tracker on new YOLO detection...")
            
            tracker.stop()
            
            if HISTORY_TRACK:
                success = initialize_tracker_with_history(tracker, 
                                                         state.last_detection_frame_id, 
                                                         state.last_detection_box)
            else:
                success = initialize_tracker_current_frame(tracker, state.last_detection_box)
            
            if success:
                state.mode = "TRACKING"
                state.stats['reinitialized'] += 1
                state.last_yolo_check_time = time.time()
                print(f"✓ Tracker reinitialized!")
            else:
                print(f"✗ Reinit failed, going back to search")
                state.mode = "SEARCHING"
        
        elif state.mode == "LOST":
            print(f"\n❌ Bird lost. Returning to SEARCHING mode...")
            tracker.stop()
            state.tracker_active = False
            state.tracker_box = None
            state.last_detection_box = None
            state.current_trajectory_dir = None
            state.mode = "SEARCHING"
            time.sleep(0.5)  # Small pause before restarting search
        
        # FPS tracking
        loop_time = time.time() - loop_start
        fps = 1.0 / loop_time if loop_time > 0 else 0
        fps_tracker.append(fps)
        state.stats['avg_tracker_fps'] = np.mean(fps_tracker)
        
        # Print status periodically
        if time.time() - last_fps_print >= 2.0:
            yolo_fps_str = f"YOLO={state.stats['avg_yolo_fps']:.1f}fps" if state.stats['avg_yolo_fps'] > 0 else "YOLO=N/A"
            print(f"\n[STATUS] Mode={state.mode} | Tracker={state.stats['avg_tracker_fps']:.1f}fps | {yolo_fps_str} | "
                  f"Frame={state.frame_counter} | History={len(state.frame_history)}")
            last_fps_print = time.time()

# ==================== MAIN ====================
def main():
    print("\n🚀 Initializing Advanced Bird Tracking System...")
    
    # Initialize camera
    print("📷 Starting camera...")
    camera = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS)
    camera.start()
    time.sleep(2.0)
    print("✓ Camera ready")
    
    # Initialize YOLO detector
    print("🧠 Loading YOLO detector...")
    detector = YoloDetector()
    print("✓ Detector ready")
    
    # Start YOLO validation worker
    yolo_thread = threading.Thread(target=yolo_worker, args=(detector,), daemon=True)
    yolo_thread.start()
    
    # Start Flask in background
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, 
                                                           threaded=True, debug=False), 
                                   daemon=True)
    flask_thread.start()
    print(f"\n🌐 Web interface: http://localhost:5000\n")
    
    time.sleep(1.0)
    
    # Run main tracking loop
    try:
        main_tracking_loop(camera, detector)
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutting down...")
        state.yolo_queue.put(None)  # Signal YOLO worker to stop
        camera.stop()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()
