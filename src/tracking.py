import cv2
import time

class ObjectTracker:
    def __init__(self, tracker_type="CSRT"):
        """
        tracker_type: "CSRT" (Genau), "MOSSE" (Schnell)
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.is_tracking = False

    def _create_tracker(self):
        tracker_name = self.tracker_type.upper()

        # VERSUCH 1: OpenCV 4.5+ Legacy API
        # if hasattr(cv2, 'legacy'):
        #     if tracker_name == "CSRT": return cv2.legacy.TrackerCSRT_create()
        #     if tracker_name == "MOSSE": return cv2.legacy.TrackerMOSSE_create()
        #     if tracker_name == "KCF": return cv2.legacy.TrackerKCF_create()

        # VERSUCH 2: Alte OpenCV API
        if tracker_name == "CSRT" and hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
        if tracker_name == "MOSSE" and hasattr(cv2, 'TrackerMOSSE_create'):
            return cv2.TrackerMOSSE_create()
        
        # Fallback
        if hasattr(cv2, 'TrackerMIL_create'):
            return cv2.TrackerMIL_create()
        
        raise AttributeError("Kein passender Tracker gefunden")

    def start(self, frame, bbox):
        """
        Initialisiert den Tracker.
        """
        x, y, w, h = [int(v) for v in bbox]
        
        # Sicherheitscheck
        if w <= 5 or h <= 5:
            self.is_tracking = False
            return

        # Optimierung: Box verkleinern (10% Rand weg), um Fokus auf Zentrum zu legen
        shrink_w = int(w * 0.10)
        shrink_h = int(h * 0.10)
        x += shrink_w
        y += shrink_h
        w -= (shrink_w * 2)
        h -= (shrink_h * 2)
        
        # Bildrand Check
        h_img, w_img = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0: return

        clean_bbox = (x, y, w, h)
        self.tracker = self._create_tracker()
        self.tracker.init(frame, clean_bbox)
        self.is_tracking = True
        print(f"🎯 Tracker gestartet: {clean_bbox}")

    def update(self, frame):
        if not self.is_tracking or self.tracker is None:
            return False, None

        success, box = self.tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            return True, (x, y, w, h)
        else:
            self.is_tracking = False
            return False, None

    def stop(self):
        self.is_tracking = False
        self.tracker = None
    
    def is_active(self):
        """Check if tracker is currently active"""
        return self.is_tracking
    
    def get_current_box(self):
        """Get current tracked bounding box (if tracking)"""
        return None  # Box is returned by update() method

    @staticmethod
    def calculate_iou(boxA, boxB):
        """
        Berechnet die Überlappung (Intersection over Union) zweier Boxen.
        Wert 0.0 = Keine Überlappung
        Wert 1.0 = Exakt gleich
        """
        # Box Format: (x, y, w, h) -> umrechnen in (x1, y1, x2, y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # Schnittfläche berechnen
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        # Fläche der beiden Boxen
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        # IoU = Schnitt / (Fläche A + Fläche B - Schnitt)
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou