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
        
        # 1. Automatische Erkennung der Input-Größe
        # EfficientDet nutzt z.B. 320x320, MobileNet 300x300
        self.input_shape = self.input_details[0]['shape'] 
        self.input_h = self.input_shape[1]
        self.input_w = self.input_shape[2]
        self.input_index = self.input_details[0]['index']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"🤖 Modell geladen. Input-Größe: {self.input_w}x{self.input_h}, Typ: {self.input_dtype}")

    def detect(self, img_crop):
        """
        Führt die Erkennung auf einem Bildausschnitt durch.
        Return: (found, score, (x, y, w, h))
        """
        if img_crop is None or img_crop.size == 0:
            return False, 0.0, None

        # 1. Resize auf die VOM MODELL geforderte Größe (dynamisch)
        input_data = cv2.resize(img_crop, (self.input_w, self.input_h))

        # 2. Input vorbereiten
        # Wenn wir nur einen Kanal haben (Grayscale), machen wir daraus 3 (Fake-RGB)
        if len(input_data.shape) == 2: 
            input_data = cv2.merge([input_data, input_data, input_data])
            
        # Farbe korrigieren: TFLite Modelle erwarten meist RGB (OpenCV liefert BGR)
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        
        # Batch-Dimension hinzufügen: [H, W, 3] -> [1, H, W, 3]
        input_data = np.expand_dims(input_data, axis=0)

        # Normalisierung (nur nötig, wenn das Modell Float erwartet)
        if self.input_dtype == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # Bei uint8 Modellen (wie EfficientDet Int8) einfach die 0-255 Werte nehmen
            input_data = np.uint8(input_data)

        # 3. Inferenz
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()

        # 4. Outputs intelligent zuordnen
        # Wir holen alle Output-Tensoren
        outputs = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        
        boxes, classes, scores = None, None, None

        for out in outputs:
            # Boxen sind immer [1, N, 4] (Koordinaten)
            if out.ndim == 3 and out.shape[2] == 4:
                boxes = out[0]
            # Scores sind [1, N] und haben oft Float-Werte
            elif out.ndim == 2 and (out.dtype == np.float32 or out.dtype == np.float16):
                scores = out[0]
            # Klassen sind [1, N]
            elif out.ndim == 2:
                classes = out[0]

        # Fallback, falls Heuristik versagt 
        if boxes is None or scores is None or classes is None:
            # Harte Zuweisung für typische EfficientDet Lite Reihenfolge:
            # Index 1 = Boxen, Index 3 = Klassen, Index 0 = Scores (kann variieren!)
            # Wir brechen hier lieber sicher ab oder loggen Warning, statt Crash.
            # Für EfficientDet Lite0 int8 ist oft: 0=Scores, 1=Boxes, 2=Count, 3=Classes
            # Wir versuchen es mit festen Indizes als letzten Versuch, falls Shapes nicht eindeutig waren:
            if len(outputs) >= 4:
                scores = outputs[0][0]
                boxes = outputs[1][0]
                # classes = outputs[3][0] # Index 3 bei manchen Versionen
                # Wir suchen den Integer Tensor für Klassen
                for i, o in enumerate(outputs):
                    if i != 0 and i != 1 and o.ndim == 2:
                        classes = o[0]
                        break

        if boxes is None:
            return False, 0.0, None

        # 5. Bestes Ergebnis suchen
        best_bird_score = 0.0
        best_box = None

        for i in range(len(scores)):
            score = float(scores[i])
            if score < MIN_CONFIDENCE:
                continue
            
            # EfficientDet nutzt COCO Labels. 
            # ID 16 ist meistens "Bird". Wir prüfen den Bereich 15-17 sicherheitshalber.
            class_id = int(classes[i])
            
            if class_id in [15, 16, 17]: 
                if score > best_bird_score:
                    best_bird_score = score
                    
                    # Box extrahieren: [ymin, xmin, ymax, xmax]
                    ymin, xmin, ymax, xmax = boxes[i]
                    
                    # Koordinaten auf den Crop hochrechnen
                    h_crop, w_crop = img_crop.shape[:2]
                    x = int(xmin * w_crop)
                    y = int(ymin * h_crop)
                    w = int((xmax - xmin) * w_crop)
                    h = int((ymax - ymin) * h_crop)
                    
                    best_box = (x, y, w, h)

        if best_box:
            return True, best_bird_score, best_box
        
        return False, 0.0, None