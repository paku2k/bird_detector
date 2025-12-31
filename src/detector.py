import cv2
import numpy as np
import os
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from config import MODEL_PATH, MIN_CONFIDENCE

class ObjectDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")

        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape'] 
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        self.input_index = self.input_details[0]['index']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"🤖 Standard-Modell geladen (EfficientDet/SSD). Input: {self.input_w}x{self.input_h}")

    def detect_all_objects(self, img, conf_threshold=MIN_CONFIDENCE):
        """
        Gibt ALLE Objekte zurück. Format kompatibel zu YoloDetector.
        Return: [(class_id, score, (x, y, w, h)), ...]
        """
        if img is None: return []

        # 1. Resize & Preprocessing
        input_data = cv2.resize(img, (self.input_w, self.input_h))

        if len(input_data.shape) == 2: 
            input_data = cv2.merge([input_data, input_data, input_data])
            
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)

        if self.input_dtype == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            input_data = np.uint8(input_data)

        # 2. Inferenz
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()

        # 3. Outputs holen
        outputs = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]

        boxes, classes, scores = None, None, None

        # Intelligentes Mapping
        for out in outputs:
            if out.ndim == 3 and out.shape[2] == 4: boxes = out[0]
            elif out.ndim == 2 and (out.dtype == np.float32 or out.dtype == np.float16): scores = out[0]
            elif out.ndim == 2: classes = out[0]

        # Fallback für EfficientDet Lite Reihenfolge
        if boxes is None and len(outputs) >= 4:
             # Oft: 0=Scores, 1=Boxes, 2=Count, 3=Classes
             # Wir probieren eine typische Reihenfolge, falls Auto-Detect fehlschlug
             scores = outputs[0][0]
             boxes = outputs[1][0]
             classes = outputs[3][0]

        if boxes is None or scores is None or classes is None:
            return []

        results = []
        h_orig, w_orig = img.shape[:2]

        for i in range(len(scores)):
            score = float(scores[i])
            if score < conf_threshold: continue
            
            ymin, xmin, ymax, xmax = boxes[i]
            
            # EfficientDet gibt normalisierte Boxen (0-1) zurück
            left = int(xmin * w_orig)
            top = int(ymin * h_orig)
            width = int((xmax - xmin) * w_orig)
            height = int((ymax - ymin) * h_orig)
            
            class_id = int(classes[i])
            
            results.append((class_id, score, (left, top, width, height)))
            
        return results

    def detect(self, img_crop):
        """Wrapper für Kompatibilität mit altem Code"""
        all_objs = self.detect_all_objects(img_crop, MIN_CONFIDENCE)
        
        best_bird_score = 0.0
        best_box = None

        for cid, score, box in all_objs:
            # ID 16 = Bird in COCO (manchmal 15/17 je nach offset)
            if cid in [15, 16, 17] and score > best_bird_score:
                best_bird_score = score
                best_box = box
        
        if best_box: return True, best_bird_score, best_box
        return False, 0.0, None