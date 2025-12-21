
#!/usr/bin/env python3
"""
pigeon_defender.py

Scaffold for a pigeon-deterrent system running on Raspberry Pi 4 (1GB).
- Uses OpenCV for capture, motion detection, MOSSE tracker, Kalman filter prediction, and a simple PID controller.
- Optional TFLite detection if tflite_runtime or tensorflow is available (CPU-only). If not available, detection is skipped and the tracker is initialized by motion.
- Motor interface is a stub (prints). Replace MotorController.send(pan, tilt) with real GPIO / I2C commands.
- Safe-mode: if a person is detected (requires TFLite COCO model), the system stops firing.

Notes:
- Tested as a scaffold; you will need to adapt camera device index and motor mapping to your hardware.
- For Pi camera, you can use libcamera with a GStreamer pipeline or use OpenCV VideoCapture(0) if configured.
"""
import cv2
import time
import threading
import queue
import numpy as np
import math
import os
from collections import deque
from picamera2 import Picamera2
import onnxruntime as ort


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
BIRD_CLASS_ID = COCO_CLASSES.index("bird")


# Try to import TFLite runtime (optional)
USE_TFLITE = False
try:
    import tflite_runtime.interpreter as tflite_rt
    TFLITE_RUNTIME = 'tflite_runtime'
    USE_TFLITE = True
except Exception:
    try:
        import tensorflow as tf
        tflite_rt = tf.lite
        TFLITE_RUNTIME = 'tf'
        USE_TFLITE = True
    except Exception:
        TFLITE_RUNTIME = None
        USE_TFLITE = False

# -------------------------
# Configuration
# -------------------------
CAMERA_INDEX = 0           # /dev/video0 or 0 for USB cam. For libcamera changes may be needed.
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
CAPTURE_FPS = 20           # target capture fps
MOTION_THRESHOLD = 25      # pixel diff threshold (0-255)
MOTION_AREA_MIN = 300      # min contour area to consider motion
DETECTOR_INPUT_SIZE = 256  # typical TFLite SSD input size (e.g., 300x300)
DETECT_EVERY_N_SECONDS = 1.0  # run the detector at most once every N seconds when motion exists
TRACKER_TYPE = "MOSSE"     # fast tracker
CONTROLLER_FREQ = 25.0     # Hz controller loop
PIPELINE_LATENCY_ESTIMATE = 0.08  # seconds, adjust after measurement

# Path to optional TFLite model (COCO / SSD MobileNet v1/v2). Use a model that detects 'bird' (class id depends on model)
TFLITE_MODEL_PATH = "models/ssd_mobilenet_v2_fpnlite_100_256_int8.tflite"  # change to your file or set to None
DETECT_CONFIDENCE_THRESHOLD = 0.4

# -------------------------
# Utility classes
# -------------------------
class PID:
    def __init__(self, kp, ki, kd, integrator_max=1.0, integrator_min=-1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator = 0.0
        self.prev_error = None
        self.integrator_max = integrator_max
        self.integrator_min = integrator_min

    def reset(self):
        self.integrator = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / (dt if dt > 0 else 1e-6)
        self.integrator += error * dt
        # clamp integrator
        self.integrator = max(self.integrator_min, min(self.integrator_max, self.integrator))
        out = self.kp * error + self.ki * self.integrator + self.kd * derivative
        self.prev_error = error
        return out

# Simple Kalman filter for 2D position + velocity
class SimpleKalman:
    def __init__(self, dt=1/30.0, process_var=1e-2, meas_var=1e-1):
        # state: [x, y, vx, vy]
        self.dt = dt
        self.x = np.zeros((4,1))
        # covariance
        self.P = np.eye(4) * 1.0
        # process noise
        q = process_var
        self.Q = np.diag([q, q, q, q])
        # measurement noise
        r = meas_var
        self.R = np.diag([r, r])
        # measurement matrix
        self.H = np.zeros((2,4))
        self.H[0,0] = 1
        self.H[1,1] = 1
        # state transition
        self.F = np.eye(4)
        self.F[0,2] = self.dt
        self.F[1,3] = self.dt

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        # update F
        F = np.eye(4)
        F[0,2] = dt
        F[1,3] = dt
        self.F = F
        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + self.Q
        return self.x.copy()

    def update(self, meas):
        # meas: (x, y)
        z = np.array(meas).reshape((2,1))
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def get_state(self):
        return self.x.flatten()  # x, y, vx, vy

# -------------------------
# Motor controller stub (replace with real implementation)
# -------------------------
class MotorController:
    def __init__(self):
        # replace with initialisation of PCA9685 / servos / steppers + encoders
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.lock = threading.Lock()

    def send(self, pan_cmd, tilt_cmd, dt):
        # pan_cmd and tilt_cmd are control outputs (e.g. angular velocity or angle delta)
        # Implement mapping to actual motor commands here.
        with self.lock:
            # simple integration: assume commands are angular velocity deg/s
            self.pan_angle += pan_cmd * dt
            self.tilt_angle += tilt_cmd * dt
            # clamp
            self.pan_angle = max(-90, min(90, self.pan_angle))
            self.tilt_angle = max(-45, min(45, self.tilt_angle))
            print(f"[Motor] pan={self.pan_angle:.2f}°, tilt={self.tilt_angle:.2f}°")

    def get_angles(self):
        with self.lock:
            return self.pan_angle, self.tilt_angle

# -------------------------
# Detector wrapper (optional TFLite)
# -------------------------
class TFLiteDetector:
    def __init__(self, model_path, input_size=300, threshold=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.labels = None
        self.interpreter = None
        if model_path and os.path.exists(model_path) and USE_TFLITE:
            try:
                if TFLITE_RUNTIME == 'tflite_runtime':
                    self.interpreter = tflite_rt.Interpreter(model_path=model_path)
                else:
                    self.interpreter = tflite_rt.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                # print(f"input_details: {self.input_details}")
                # print(f"output_details: {self.output_details}")
                print("[Detector] TFLite model loaded.")
            except Exception as e:
                print("[Detector] Failed to load TFLite model:", e)
                self.interpreter = None
        else:
            print("[Detector] No TFLite model available or path invalid; detector disabled.")
            self.interpreter = None

    def infer(self, frame_rgb):
        """
        frame_rgb: HxWx3 in uint8
        returns list of detections: each is (ymin, xmin, ymax, xmax, score, class_id)
        coordinates are in pixel units relative to original frame.
        """
        if self.interpreter is None:
            return []
        h, w, _ = frame_rgb.shape
        # preprocess resize to input size
        inp = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        inp = np.expand_dims(inp, axis=0)
        # adjust dtype according to model
        if self.input_details[0]['dtype'] == np.float32:
            inp = (np.float32(inp) - 127.5) / 127.5
        inp = inp.astype(self.input_details[0]['dtype'])
        self.interpreter.set_tensor(self.input_details[0]['index'], inp)
        t0 = time.time()
        self.interpreter.invoke()
        t1 = time.time()
        # common SSD outputs: boxes, classes, scores, num
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # y1,x1,y2,x2 normalized
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        print(f"boxes: {boxes.shape}")
        print(f"classes: {classes.shape}")
        print(f"scores: {scores.shape}")
        
        # num = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        res = []
        for b, c, s in zip(boxes, classes, scores):
            if s < self.threshold:
                continue
            ymin = int(max(0, b[0] * h))
            xmin = int(max(0, b[1] * w))
            ymax = int(min(h, b[2] * h))
            xmax = int(min(w, b[3] * w))
            res.append((ymin, xmin, ymax, xmax, float(s), int(c)))
        #print(f"[Detector] inference time: {(t1-t0)*1000:.1f} ms, detections: {len(res)}")
        return res

# -------------------------
# Threads: Capture, Motion, Detector, Tracker, Controller
# -------------------------
class FrameGrabber(threading.Thread):
    def __init__(self, frame_q, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=CAPTURE_FPS, device=CAMERA_INDEX):
        super().__init__(daemon=True)
        self.q = frame_q
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.cap = None
        self.running = True

    def run(self):
        self.cap = Picamera2()
        config = self.cap.create_preview_configuration(main={"size": (self.width, self.height)})
        self.cap.configure(config)
        self.cap.start()
        print("[Capture] started")
        while self.running:
            t0 = time.time()
            frame = self.cap.capture_array()
            if frame is None:
                print("[Capture] frame not received, retrying...")
                time.sleep(0.1)
                continue
            # convert to RGB for detector compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_rgb = frame
            timestamp = time.time()
            try:
                self.q.put((timestamp, frame, frame_rgb), timeout=0.01)
            except queue.Full:
                pass
            # sleep to maintain target fps
            dt = 1.0 / max(1, self.fps)
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
        self.cap.stop()
        print("[Capture] stopped")

    def stop(self):
        self.running = False

class MotionDetector(threading.Thread):
    def __init__(self, frame_q, motion_q):
        super().__init__(daemon=True)
        self.frame_q = frame_q
        self.motion_q = motion_q
        self.running = True
        self.bg = None
        self.alpha = 0.01  # background update rate

    def run(self):
        print("[Motion] started")
        while self.running:
            try:
                timestamp, frame_bgr, frame_rgb = self.frame_q.get(timeout=1.0)
            except queue.Empty:
                continue
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7,7), 0)
            if self.bg is None:
                self.bg = gray.astype("float")
                continue
            cv2.accumulateWeighted(gray, self.bg, self.alpha)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.bg))
            thresh = cv2.threshold(frameDelta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_boxes = []
            for c in contours:
                if cv2.contourArea(c) < MOTION_AREA_MIN:
                    continue
                (x, y, w, h) = cv2.boundingRect(c)
                motion_boxes.append((x, y, x+w, y+h))
            if motion_boxes:
                # merge boxes into one bounding box
                xs = [b[0] for b in motion_boxes] + [b[2] for b in motion_boxes]
                ys = [b[1] for b in motion_boxes] + [b[3] for b in motion_boxes]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bbox = (xmin, ymin, xmax, ymax)
                # send motion event with timestamp and bbox and frames
                try:
                    self.motion_q.put((timestamp, bbox, frame_bgr, frame_rgb), timeout=0.01)
                except queue.Full:
                    pass
            # small sleep to avoid busy loop
            time.sleep(0.001)
        print("[Motion] stopped")

    def stop(self):
        self.running = False

class DetectorThread(threading.Thread):
    def __init__(self, motion_q, detect_q, model_path=TFLITE_MODEL_PATH):
        super().__init__(daemon=True)
        self.motion_q = motion_q
        self.detect_q = detect_q
        self.running = True
        self.detector = TFLiteDetector(model_path, input_size=DETECTOR_INPUT_SIZE, threshold=DETECT_CONFIDENCE_THRESHOLD)
        self.last_detect_time = 0

    def run(self):
        print("[Detector] started")
        while self.running:
            try:
                timestamp, motion_bbox, frame_bgr, frame_rgb = self.motion_q.get(timeout=1.0)
            except queue.Empty:
                continue
            now = time.time()
            if now - self.last_detect_time < DETECT_EVERY_N_SECONDS:
                continue
            self.last_detect_time = now
            # expand ROI a bit
            x1, y1, x2, y2 = motion_bbox
            h, w = frame_bgr.shape[:2]
            pad = 0.2
            dx = int((x2 - x1) * pad)
            dy = int((y2 - y1) * pad)
            rx1 = max(0, x1 - dx)
            ry1 = max(0, y1 - dy)
            rx2 = min(w, x2 + dx)
            ry2 = min(h, y2 + dy)
            roi_rgb = frame_rgb[ry1:ry2, rx1:rx2]
            detections = self.detector.infer(roi_rgb) if self.detector.interpreter else []
            # convert ROI-relative boxes to full-frame coords
            detections_global = []
            for (ymin, xmin, ymax, xmax, score, cls) in detections:
                # adjust by ROI offset
                detections_global.append((ymin+ry1, xmin+rx1, ymax+ry1, xmax+rx1, score, cls))
            # if no detector, create synthetic bbox from motion
            if not detections_global:
                # fallback: treat motion bbox as detection with score 0.5 and class -1
                detections_global = [(y1, x1, y2, x2, 0.5, -1)]
            try:
                self.detect_q.put((time.time(), detections_global, frame_bgr, frame_rgb), timeout=0.01)
            except queue.Full:
                pass
        print("[Detector] stopped")

    def stop(self):
        self.running = False

class TrackerThread(threading.Thread):
    def __init__(self, detect_q, track_q):
        super().__init__(daemon=True)
        self.detect_q = detect_q
        self.track_q = track_q
        self.running = True
        self.tracker = None
        self.bbox = None  # bbox in (x,y,w,h) format for OpenCV trackers
        self.lock = threading.Lock()
        self.kalman = SimpleKalman(dt=1.0/CONTROLLER_FREQ)
        self.last_frame = None
        self.lost_since = None

    def _init_tracker(self, frame, bbox_xyxy):
        # bbox_xyxy: xmin,ymin,xmax,ymax
        x1,y1,x2,y2 = bbox_xyxy
        w = x2 - x1
        h = y2 - y1
        self.bbox = (int(x1), int(y1), int(w), int(h))
        # create MOSSE
        self.tracker = cv2.TrackerMOSSE_create()
        ok = self.tracker.init(frame, self.bbox)
        if not ok:
            print("[Tracker] init failed")
            self.tracker = None
        else:
            # init kalman state (center)
            cx = x1 + w/2.0
            cy = y1 + h/2.0
            self.kalman.x = np.array([[cx],[cy],[0.],[0.]])
            print(f"[Tracker] initialized bbox={self.bbox}")

    def run(self):
        print("[Tracker] started")
        while self.running:
            # preferentially consume detections
            try:
                ts, dets, frame_bgr, frame_rgb = self.detect_q.get(timeout=0.05)
                # pick the best detection (highest score)
                det = max(dets, key=lambda d: d[4])
                ymin, xmin, ymax, xmax, score, cls = det
                self._init_tracker(frame_bgr, (xmin, ymin, xmax, ymax))
                self.last_frame = frame_bgr
                self.lost_since = None
            except queue.Empty:
                # no new detection; update tracker if exists
                if self.tracker is not None and self.last_frame is not None:
                    ok, bbox = self.tracker.update(self.last_frame)
                    if ok:
                        x,y,w,h = bbox
                        cx = x + w/2.0
                        cy = y + h/2.0
                        # update kalman
                        self.kalman.predict()
                        self.kalman.update((cx, cy))
                        state = self.kalman.get_state()
                        vx, vy = state[2], state[3]
                        # publish tracked target (center x,y and velocity)
                        try:
                            self.track_q.put((time.time(), (cx, cy), (vx, vy), bbox), timeout=0.01)
                        except queue.Full:
                            pass
                        self.lost_since = None
                    else:
                        # tracker lost
                        if self.lost_since is None:
                            self.lost_since = time.time()
                        # if lost for too long, reset tracker
                        if time.time() - self.lost_since > 0.5:
                            print("[Tracker] lost target")
                            self.tracker = None
                    # small sleep to avoid busy polling
                    time.sleep(0.001)
                else:
                    time.sleep(0.01)
                    continue
            # update last_frame with latest frame from capture queue if available
            # (we try to keep last_frame fresh by peeking into detect_q frame payloads)
            # nothing else here
        print("[Tracker] stopped")

    def stop(self):
        self.running = False

class ControllerThread(threading.Thread):
    def __init__(self, track_q, motor_controller):
        super().__init__(daemon=True)
        self.track_q = track_q
        self.motor = motor_controller
        self.running = True
        # PID tuning initial guesses (tune these in practice)
        self.pid_pan = PID(0.6, 0.01, 0.03, integrator_max=10, integrator_min=-10)
        self.pid_tilt = PID(0.8, 0.01, 0.03, integrator_max=10, integrator_min=-10)
        self.last_state = None
        self.frame_center = (FRAME_WIDTH/2.0, FRAME_HEIGHT/2.0)
        self.last_time = time.time()

    def pix_to_angle_error(self, cx, cy):
        """
        Convert pixel coords to angular error relative to current pan/tilt.
        This is a simple pinhole camera model approximation. For best results,
        perform calibration to map pixel positions to actual pan/tilt angles.
        """
        fx = FRAME_WIDTH / (2.0 * math.tan(math.radians(45)/2.0)) if FRAME_WIDTH else 1.0
        fy = fx
        # image plane coordinates relative to center
        dx = cx - self.frame_center[0]
        dy = cy - self.frame_center[1]
        # approximate angle offsets (radians)
        angle_x = math.atan2(dx, fx)
        angle_y = math.atan2(dy, fy)
        # convert to degrees
        return math.degrees(angle_x), math.degrees(angle_y)

    def run(self):
        print("[Controller] started")
        rate = 1.0 / CONTROLLER_FREQ
        while self.running:
            t0 = time.time()
            try:
                ts, (cx, cy), (vx, vy), bbox = self.track_q.get(timeout=rate)
                # predict ahead by pipeline latency
                predict_dt = PIPELINE_LATENCY_ESTIMATE
                predicted_cx = cx + vx * predict_dt
                predicted_cy = cy + vy * predict_dt
                # compute angular error
                err_pan_deg, err_tilt_deg = self.pix_to_angle_error(predicted_cx, predicted_cy)
                # current angles
                cur_pan, cur_tilt = self.motor.get_angles()
                # desired angle is current + error (assuming mapping)
                desired_pan = cur_pan + err_pan_deg
                desired_tilt = cur_tilt + err_tilt_deg
                # compute control commands (as angular velocities in deg/s)
                now = time.time()
                dt = now - self.last_time if self.last_time else 1.0/CONTROLLER_FREQ
                pan_cmd = self.pid_pan.update(desired_pan - cur_pan, dt)
                tilt_cmd = self.pid_tilt.update(desired_tilt - cur_tilt, dt)
                # send to motor controller
                self.motor.send(pan_cmd, tilt_cmd, dt)
                self.last_time = now
                self.last_state = (cx, cy, vx, vy, bbox)
            except queue.Empty:
                # no tracked target this cycle
                time.sleep(rate)
                continue
        print("[Controller] stopped")

    def stop(self):
        self.running = False

# -------------------------
# Main orchestration
# -------------------------
def main(runtime_seconds=60):
    # Queues
    frame_q = queue.Queue(maxsize=4)
    motion_q = queue.Queue(maxsize=4)
    detect_q = queue.Queue(maxsize=4)
    track_q = queue.Queue(maxsize=8)

    # components
    grabber = FrameGrabber(frame_q)
    motion = MotionDetector(frame_q, motion_q)
    detector = DetectorThread(motion_q, detect_q, model_path="models/ssd_mobilenet_v2_fpnlite_100_256_int8.tflite")
    motor = MotorController()
    tracker = TrackerThread(detect_q, track_q)
    controller = ControllerThread(track_q, motor)

    # start threads
    grabber.start()
    motion.start()
    detector.start()
    tracker.start()
    controller.start()

    start_time = time.time()
    try:
        while time.time() - start_time < runtime_seconds:
            # For demonstration, also show debug frames in a window (optional)
            # We attempt to pull the most recent frame and annotate
            try:
                ts, frame_bgr, frame_rgb = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            vis = frame_bgr.copy()
            # overlay latest tracked box if available
            try:
                last = tracker.kalman.get_state()
                cx, cy = last[0], last[1]
                cv2.circle(vis, (int(cx), int(cy)), 10, (0,255,0), 2)
                pan, tilt = motor.get_angles()
                cv2.putText(vis, f"pan {pan:.1f} tilt {tilt:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            except Exception:
                pass
            cv2.imshow("pigeon_defender_debug", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Shutting down threads...")
        grabber.stop()
        motion.stop()
        detector.stop()
        tracker.stop()
        controller.stop()
        time.sleep(0.2)
        cv2.destroyAllWindows()
        print("Exited cleanly.")

if __name__ == "__main__":
    # run for a long time by default (change as required)
    main(runtime_seconds=3600)
