Projekt: "BirdDefense" auf Raspberry Pi 4 (1GB RAM) mit Servos und Wasserpistole. Ziel: Vögel erkennen, tracken und vertreiben. Herausforderung: Das System muss effizient sein (1GB RAM) und auch funktionieren, wenn der Vogel landet und stillsitzt.

Hardware:

RPi 4 (1GB), Pi Camera V2 (NoIR).

Input-Bild ist zwingend Grayscale, um Rotstich der NoIR-Kamera zu entfernen.

Servos für Pan/Tilt/Trigger.

Bitte erstelle die Software-Architektur und den Python-Code (in Modulen) basierend auf dieser Logik:

Modul 1: VisionEngine
Da Vögel klein sind, brauchen wir einen Zoom. Da sie stillsitzen können, brauchen wir zwei Such-Modi.

Funktion find_bird(frame): Implementiere eine zweistufige Suchstrategie:

Strategie A: Motion-Guided (Priorität)

Prüfe mit cv2.absdiff, ob Bewegung vorliegt.

Wenn JA: Schneide die Bounding Box der Bewegung (+ Padding) aus (Dynamic ROI).

Führe TFLite Inferenz (MobileNet V2 SSD) auf diesem Crop durch.

Strategie B: Grid-Scan (Fallback)

Wenn KEINE Bewegung erkannt wird (aber der PIR Sensor ausgelöst hat), unterteile das Bild in ein 2x2 Raster (4 Quadranten).

Führe die TFLite Inferenz nacheinander auf diesen 4 festen Ausschnitten durch.

Dies stellt sicher, dass auch ein bereits sitzender Vogel gefunden wird.

Preprocessing:

Input immer Grayscale.

Für TFLite: Stacke den Grayscale-Kanal zu einem 3-Kanal-Bild (np.dstack), da MobileNet RGB erwartet.

Modul 2: TrackingSystem (Die Lösung für sitzende Vögel)
Sobald VisionEngine einen Vogel findet, darf nicht mehr neu gesucht werden (zu langsam/unsicher).

Initialisiere sofort einen OpenCV Tracker (cv2.TrackerCSRT_create() oder MOSSE für Speed).

Der Tracker übernimmt die Kontrolle. Er verfolgt das visuelle Muster (Texture) des Vogels.

Vorteil: Wenn der Vogel landet und stillsitzt, behält der Tracker ihn im Visier (im Gegensatz zum Motion Detector).

Führe alle 30 Frames eine Re-Validierung durch (Ein einzelner TFLite-Check auf der Tracker-Position), um sicherzugehen, dass wir noch einen Vogel tracken und nicht den Hintergrund.

Modul 3: Ballistics & ServoControl
Rechne die Koordinaten vom Tracker (X, Y im Bild) in Servo-Winkel um.

Nutze einen PID-Regler, um die Kamera sanft auf den Vogel auszurichten.

Gravity Compensation: Wenn der Vogel weit weg ist (Bounding Box klein), ziele etwas höher (Pitch + Offset).

Modul 4: MainLoop (State Machine)
WAIT: Warte auf PIR.

SEARCH: Hole Frame. Rufe VisionEngine.find_bird() auf (probiert erst Motion, dann Grid).

LOCK: Vogel gefunden -> Initialisiere Tracker.

TRACK & FIRE:

Update Tracker.

Update Servos (PID).

Wenn Ziel im Fadenkreuz (Error < Threshold) -> fire_gun().

LOST: Wenn Tracker fehlschlägt -> Zurück zu SEARCH.

Code-Anforderungen:

Nutze tflite_runtime.

Achte auf Threading: Das Kamerabild und die Servos müssen flüssig laufen, auch wenn die Inferenz (Grid Scan) mal 500ms dauert.

Erstelle eine config.py für PID-Werte und Pins.