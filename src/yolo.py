import cv2
import numpy as np
import os
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from config import MODEL_PATH, MIN_CONFIDENCE

class YoloDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO Modell nicht gefunden: {MODEL_PATH}")

        print(f"🚀 Lade YOLO11 Modell: {os.path.basename(MODEL_PATH)}")
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_idx = self.input_details[0]['index']
        self.output_idx = self.output_details[0]['index']
        
        # Automatische Größen-Erkennung (z.B. 640x640)
        self.input_shape = self.input_details[0]['shape'] 
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        
        # Datentyp prüfen (Int8 oder Float)
        print(f"Input_details_dtype {self.input_details[0]['dtype']}")
        self.is_quantized = (self.input_details[0]['dtype'] != np.float32)
        
        if self.is_quantized:
            self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
            self.output_scale, self.output_zero_point = self.output_details[0]['quantization']
            print(f"ℹ️  Modell ist Quantisiert (Int8). Input: {self.input_w}x{self.input_h}")
        else:
            print(f"ℹ️  Modell ist Float32. Input: {self.input_w}x{self.input_h}")

    def _preprocess(self, img):
        """Hilfsfunktion: Bild aufbereiten (Resize, Color-Conversion, Normierung)"""
        # Resize auf Modell-Input-Größe
        input_img = cv2.resize(img, (self.input_w, self.input_h))
        
        # Automatische Erkennung: Grayscale oder Farbe?
        # YOLO erwartet RGB.
        if len(input_img.shape) == 2: 
            # Input ist Grayscale -> Konvertiere Gray zu RGB
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
        else:
            # Input ist BGR (OpenCV Standard) -> Konvertiere BGR zu RGB
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalisierung
        if self.is_quantized:
            input_data = (input_img / 255.0) / self.input_scale + self.input_zero_point
            input_data = input_data.astype(self.input_details[0]['dtype'])
        else:
            input_data = (input_img / 255.0).astype(np.float32)
            
        return np.expand_dims(input_data, axis=0)

    def detect_all_objects(self, img, conf_threshold=MIN_CONFIDENCE):
        """
        Gibt ALLE erkannten Objekte zurück (ungefiltert).
        Return: Liste von (class_id, score, (x, y, w, h))
        """
        if img is None: return []

        # 1. Inferenz
        input_data = self._preprocess(img)
        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()

        # 2. Output holen & Aufbereiten
        output_data = self.interpreter.get_tensor(self.output_idx)[0]
        
        if self.is_quantized:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

        # Transponieren [84, N] -> [N, 84]
        if output_data.shape[0] < output_data.shape[1]:
            output_data = output_data.T

        boxes = []
        confidences = []
        class_ids = []
        
        # Koordinaten-Check (Normalisiert vs. Absolute Pixel)
        max_val = np.max(output_data[:, 0:4])
        is_normalized = max_val <= 1.1

        # 3. Decoding Loop
        for det in output_data:
            classes_scores = det[4:]
            class_id = np.argmax(classes_scores)
            score = classes_scores[class_id]
            
            if score < conf_threshold:
                continue

            cx, cy, w, h = det[0:4]
            
            if is_normalized:
                cx *= self.input_w
                cy *= self.input_h
                w *= self.input_w
                h *= self.input_h
            
            # Skalierung auf Originalbild
            orig_h, orig_w = img.shape[:2]
            x_factor = orig_w / self.input_w
            y_factor = orig_h / self.input_h
            
            left = int((cx - w/2) * x_factor)
            top = int((cy - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(score))
            class_ids.append(class_id)

        if not boxes:
            return []

        # 4. NMS (Klassen-Agnostisch für Einfachheit, oder wir vertrauen YOLO11's Trennung)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
        
        results = []
        if len(indices) > 0:
            for i in indices:
                # cv2.dnn.NMSBoxes gibt manchmal verschachtelte Listen zurück
                idx = i if isinstance(i, (int, np.integer)) else i[0]
                results.append((class_ids[idx], confidences[idx], tuple(boxes[idx])))
        
        return results

    def detect(self, img):
        """
        Original-Methode (Kompatibilitätsmodus):
        Gibt nur den BESTEN VOGEL zurück.
        Return: (found, score, box)
        """
        all_detections = self.detect_all_objects(img, MIN_CONFIDENCE)
        
        best_bird_score = 0.0
        best_bird = None

        for class_id, score, box in all_detections:
            # Filter auf Vögel (COCO ID 14, 15, 16)
            if class_id in [14, 15, 16]:
                if score > best_bird_score:
                    best_bird_score = score
                    best_bird = box
        
        if best_bird:
            return True, best_bird_score, best_bird
            
        return False, 0.0, None