import time
import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
from libcamera import controls

# ----------------- Konfiguration -----------------
MODELS = [
   {"path": "models/yolo11n_640_nms.onnx", "size": (640, 640)},
#    {"path": "models/yolov8n_320_nms.onnx", "size": (320, 320)},
]
CAMERA_SIZE = (640, 480)       # Kamera-Ausgabe (Breite, Hoehe)
FRAMES_TO_TEST = 50          # Anzahl Frames pro Test
FRAME_SKIP = 1                # nur jedes N-te Frame verarbeiten
CONF_THRESHOLD = 0.25          # YOLO Confidence
IOU_THRESHOLD = 0.45          # IOU-Threshold fuer Non-Max Suppression (obwohl bei NMS-Modell im ONNX-Graph nicht direkt verwendet)

# --- NEUE OPTIONEN ---
SHOW_PREVIEW = True           # Option zum Anzeigen des Vorschaubilds
PREVIEW_SIZE = (800, 800)   # Groesse des Vorschaubilds
USE_GRAYSCALE = False         # Option, um Graustufenbild zu verwenden
USE_COLOR_CORRECTION = False  # Option, um eine Farbkorrektur anzuwenden
DETECT_BIRDS_ONLY = True      # NEU: Option, nur Voegel zu erkennen

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

# Die ID fuer die Klasse "bird" in der COCO-Klassenliste ist 14
BIRD_CLASS_ID = COCO_CLASSES.index("bird")

# ----------------- Funktionen -----------------
def correct_red_tint(image):
    # Einfache Farbkorrektur, um Rotstich zu entfernen
    # Reduziert den Rotkanal und erhoeht Gruen und Blau
    b, g, r = cv2.split(image)
    r_corrected = r * 0.9
    g_corrected = g * 1.1
    b_corrected = b * 1.1
    r_corrected = r_corrected.clip(0, 255).astype("uint8")
    g_corrected = g_corrected.clip(0, 255).astype("uint8")
    b_corrected = b_corrected.clip(0, 255).astype("uint8")
    corrected_image = cv2.merge([b_corrected, g_corrected, r_corrected])
    return corrected_image

# ----------------- Kamera Setup -----------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": CAMERA_SIZE, "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1) # Kamera stabilisieren

# ----------------- Performance-Test fuer alle Modelle -----------------
for model_cfg in MODELS:
    model_path = model_cfg["path"]
    YOLO_INPUT_SIZE = model_cfg["size"]
    print(f"\n=== Test mit Modell {model_path}, Inputgroesse {YOLO_INPUT_SIZE} ===")

    # ONNX-Session laden
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"input_name: {input_name}")
    
    input_shape = session.get_inputs()[0].shape
    print(f"input_shape: {input_shape}")
    
    if isinstance(input_shape[2], str) or isinstance(input_shape[3], str):
        print(f"WARNUNG: ONNX-Modell hat dynamische Input-Dimensionen. Verwende konfigurierte Groesse: {YOLO_INPUT_SIZE}")
        input_height, input_width = YOLO_INPUT_SIZE
    else:
        input_height = input_shape[2]
        input_width = input_shape[3]
        if (input_width, input_height) != YOLO_INPUT_SIZE:
            print(f"WARNUNG: YOLO_INPUT_SIZE ({YOLO_INPUT_SIZE}) stimmt nicht mit ONNX-Modell-Input ({input_width, input_height}) ueberein.")
            print(f"Verwende Input-Groesse des ONNX-Modells: {(input_width, input_height)}")
            YOLO_INPUT_SIZE = (input_width, input_height)

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    if SHOW_PREVIEW:
        cv2.namedWindow("YOLO Live Vorschau", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Live Vorschau", PREVIEW_SIZE[0], PREVIEW_SIZE[1])

    while processed_count < FRAMES_TO_TEST:
        frame_count += 1
        frame = picam2.capture_array()
        if frame is None:
            continue

        if USE_COLOR_CORRECTION:
            frame = correct_red_tint(frame)

        if USE_GRAYSCALE:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            img_input_raw = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        else:
            img_input_raw = frame.copy()

        if frame_count % FRAME_SKIP != 0:
            if SHOW_PREVIEW:
                cv2.imshow("YOLO Live Vorschau", cv2.resize(frame, PREVIEW_SIZE))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        original_h, original_w, _ = frame.shape
        
        # ----------------- Preprocessing mit Padding -----------------
        ratio_w = YOLO_INPUT_SIZE[0] / original_w
        ratio_h = YOLO_INPUT_SIZE[1] / original_h
        ratio = min(ratio_w, ratio_h)
        
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        
        pad_w = YOLO_INPUT_SIZE[0] - new_w
        pad_h = YOLO_INPUT_SIZE[1] - new_h
        
        img_resized = cv2.resize(img_input_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        top = int(pad_h / 2)
        bottom = pad_h - top
        left = int(pad_w / 2)
        right = pad_w - left
        
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img_input = img_padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        
        # ----------------- Inference -----------------
        outputs = session.run(None, {input_name: img_input})
        print(f"outputs.type: {type(outputs)}, outputs[0].shape: {outputs[0].shape}")
        
        results = outputs[0][0, outputs[0][0, :, 4] > CONF_THRESHOLD]
        
        # ----------------- Ergebnisse zeichnen -----------------
        display_frame = img_padded.copy()
        detected_classes = []

        for r in results:
            x1, y1, x2, y2, conf, class_id_float = r
            class_id = int(class_id_float)
            
            # NEU: Pruefe, ob nur Voegel erkannt werden sollen
            if DETECT_BIRDS_ONLY and class_id != BIRD_CLASS_ID:
                continue

            x1_scaled = int(x1)
            y1_scaled = int(y1)
            x2_scaled = int(x2)
            y2_scaled = int(y2)
            
            x1_scaled = max(0, x1_scaled)
            y1_scaled = max(0, y1_scaled)
            x2_scaled = min(YOLO_INPUT_SIZE[0], x2_scaled)
            y2_scaled = min(YOLO_INPUT_SIZE[1], y2_scaled)
            
            class_name = COCO_CLASSES[class_id]
            detected_classes.append(class_name)

            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
            label = f"{class_name} {conf:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            cv2.putText(display_frame, label, (x1_scaled, y1_scaled - 10), font, font_scale, color, font_thickness)

        if detected_classes:
            print(f"Frame {processed_count+1}: erkannte Objekte -> {', '.join(detected_classes)}")
        else:
            print(f"Frame {processed_count+1}: keine Objekte erkannt")

        # ----------------- Vorschau anzeigen -----------------
        if SHOW_PREVIEW:
            preview_display = cv2.resize(display_frame, PREVIEW_SIZE, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("YOLO Live Vorschau", preview_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        processed_count += 1

    # ----------------- Performance -----------------
    elapsed = time.time() - start_time
    fps = processed_count / elapsed
    print(f"\nFrames gelesen: {frame_count}")
    print(f"Frames verarbeitet: {processed_count}")
    print(f"Zeit: {elapsed:.2f}s, FPS: {fps:.2f}")

# Aufraeumen
picam2.stop()
cv2.destroyAllWindows()
