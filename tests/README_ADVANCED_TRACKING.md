# Advanced Bird Tracking Test

Comprehensive bird tracking system with history-based initialization, background YOLO validation, and live web visualization.

## Features

### 1. **Continuous Bird Detection**
- YOLO detector continuously scans frames for birds
- Activates tracker immediately upon detection

### 2. **Smart Tracker Initialization**
Two modes controlled by `HISTORY_TRACK` setting:

#### Mode 1: History Tracking (`HISTORY_TRACK=1`)
- Tracker initializes on the frame where YOLO detected the bird
- Automatically tracks through all buffered frames up to present
- Ensures tracker is synchronized even if YOLO took 1+ seconds
- Best for: Maximum accuracy, handling detection lag

#### Mode 2: Current Frame (`HISTORY_TRACK=0`)
- Tracker initializes on the most recent frame
- Faster startup, but bird may have moved since detection
- Best for: Fast-moving scenarios, minimal lag systems

### 3. **Background YOLO Validation**
- Runs YOLO periodically (every `YOLO_INTERVAL` seconds) in background thread
- Doesn't block tracker (tracker runs at full speed)
- Calculates IOU between YOLO detection and tracker position
- Auto-reinitializes tracker if IOU drops below threshold
- Aborts tracking if YOLO loses bird completely

### 4. **Optimized Bounding Box**
- Shrinks detection bbox by `SHRINK_BBOX_PERCENT` on each side
- Focuses tracker on bird body features only (not background)
- Reduces tracker drift from background movement

### 5. **Live Web Visualization**
- Flask web server on port 5000
- Real-time video stream with annotations:
  - Red box = YOLO detection
  - Green box = Tracker position
  - Crosshair on tracked center point
- Status overlay with mode, FPS, frame count
- Statistics: detections, tracks, losses, reinitializations
- YOLO inference time display

### 6. **Frame History Buffer**
- Circular buffer keeps last N frames in memory
- Each frame tagged with ID and timestamp
- Enables history-based tracker initialization
- Configurable size (`FRAME_HISTORY_SIZE`)

## Usage

### Basic Run
```bash
cd tests
python3 test_advanced_tracking.py
```

Then open browser: `http://localhost:5000`

### Configuration

Edit settings at top of `test_advanced_tracking.py`:

```python
TRACKER_TYPE = "MOSSE"          # "MOSSE" (Fast) or "CSRT" (Accurate)
HISTORY_TRACK = 1               # 1=History, 0=Current frame
YOLO_INTERVAL = 2.0             # Seconds between validations
IOU_THRESHOLD = 0.3             # Min IOU for valid tracking
FRAME_HISTORY_SIZE = 60         # History buffer size
SHRINK_BBOX_PERCENT = 0.15      # Bbox shrink amount
```

### Presets

Use `tracking_config.py` for quick presets:

```python
# Fast tracking (MOSSE, no history, less frequent checks)
preset_fast()

# Accurate tracking (CSRT, with history, frequent checks)
preset_accurate()

# Paranoid mode (CSRT, very frequent YOLO validation)
preset_paranoid()
```

## State Machine

```
SEARCHING → (bird detected) → TRACKING
    ↑                              ↓
    └───────← (bird lost) ←────────┘
                   ↓
            REINIT_NEEDED
                   ↓
            (reinitialized)
                   ↓
              TRACKING
```

### States:
- **SEARCHING**: YOLO continuously scans for birds
- **TRACKING**: Tracker active, YOLO validates periodically
- **REINIT_NEEDED**: Tracker drift detected, needs reinitialization
- **LOST**: Bird completely lost, return to searching

## Output

### Console Output
```
🐦 BIRD DETECTED!
   Frame ID: 1234
   Confidence: 0.852
   Box: (320, 240, 80, 60)

🔄 Initializing tracker with HISTORY tracking...
   Detection was at frame_id=1234
   Current frame_id=1245
   Original bbox: (320, 240, 80, 60)
   Shrunk bbox (focus on body): (332, 252, 56, 48)
   Tracking through 11 history frames...
   ✓ Successfully tracked through 11 history frames!
   Tracker is now at current frame

✓ Tracker activated!

⏰ Time for YOLO validation (last check 2.1s ago)
  ✓ YOLO validation: Bird detected (score=0.83, time=245ms)
    IOU with tracker: 0.752 (threshold=0.3)
    ✓ Tracker still valid

[STATUS] Mode=TRACKING | FPS=28.3 | Frame=1567 | History=60
```

### Web Interface
- Live video with color-coded bounding boxes
- Mode indicator (green=tracking, orange=searching, red=lost)
- Real-time statistics
- FPS and inference time metrics

## Performance Tips

### For Raspberry Pi 4 (1GB)

**Fast Mode** (30+ FPS tracking):
```python
TRACKER_TYPE = "MOSSE"
HISTORY_TRACK = 0
YOLO_INTERVAL = 3.0
```

**Accurate Mode** (20+ FPS tracking):
```python
TRACKER_TYPE = "CSRT"
HISTORY_TRACK = 1
YOLO_INTERVAL = 2.0
```

**Balanced Mode** (25+ FPS):
```python
TRACKER_TYPE = "MOSSE"
HISTORY_TRACK = 1
YOLO_INTERVAL = 2.5
```

## Technical Details

### Threading Architecture
- **Main Thread**: Camera capture, tracker update, visualization
- **YOLO Worker Thread**: Background YOLO validation (non-blocking)
- **Flask Thread**: Web server for visualization

### Frame History
- Thread-safe circular buffer
- Each frame stored with metadata (ID, timestamp)
- Automatic cleanup of old frames
- O(1) lookup by frame ID

### IOU Calculation
```
IOU = Intersection Area / Union Area
IOU >= threshold → Tracker valid
IOU < threshold → Reinitialize tracker
```

### Bbox Shrinking
Reduces bbox by percentage on all sides:
```
x_new = x + (w * shrink%)
y_new = y + (h * shrink%)
w_new = w * (1 - 2*shrink%)
h_new = h * (1 - 2*shrink%)
```

## Dependencies

Already in `requirements.txt`:
- OpenCV (cv2)
- NumPy
- Flask
- Picamera2
- TFLite Runtime

## Troubleshooting

**Tracker keeps losing bird:**
- Increase `IOU_THRESHOLD` (try 0.2 or 0.15)
- Decrease `YOLO_INTERVAL` (more frequent checks)
- Try `HISTORY_TRACK=1` for better initialization

**System too slow:**
- Use `TRACKER_TYPE="MOSSE"` (faster)
- Increase `YOLO_INTERVAL` (less frequent validation)
- Reduce `FRAME_HISTORY_SIZE` (less memory)
- Use smaller YOLO model (320px instead of 640px)

**Tracker drifts to background:**
- Increase `SHRINK_BBOX_PERCENT` (focus more on center)
- Decrease `YOLO_INTERVAL` (catch drift sooner)
- Use `TRACKER_TYPE="CSRT"` (more accurate)

**Web interface laggy:**
- Normal - Flask streams at max ~30 FPS
- Tracker still runs at full speed in background

## Advanced Usage

### Custom YOLO Model
```python
detector = YoloDetector(model_path="models/custom_model.tflite")
```

### Different Tracker Types
Supported: `"MOSSE"`, `"CSRT"`, `"KCF"`, `"MIL"`

### Frame History Analysis
```python
# Get specific frame
frame = get_frame_by_id(1234)

# Get all frames since detection
history = get_frames_since(detection_frame_id)
```

## Statistics Tracked

- `detections`: Total YOLO bird detections
- `tracks`: Number of times tracker was activated
- `lost_tracks`: Times tracker lost bird
- `reinitialized`: Times tracker was reinitialized due to drift
- `avg_tracker_fps`: Average tracker loop FPS
- `avg_yolo_ms`: Average YOLO inference time

## Files Modified

- `src/yolo.py`: Added `model_path` parameter, `frame_id` support
- `src/tracking.py`: Added `is_active()` helper method
- `tests/test_advanced_tracking.py`: Main test script
- `tests/tracking_config.py`: Configuration helper

Enjoy tracking! 🐦🎯
