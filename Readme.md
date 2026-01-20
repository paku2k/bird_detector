# TAS: Tauben Abwehr System

**TAS** ist ein automatisiertes Vogelabwehrsystem auf Basis eines Raspberry Pi 4. Es nutzt Computer Vision und Machine Learning, um Vögel zu erkennen, zu verfolgen und mittels einer Wasserpistole zu vertreiben.

Das System ist speziell für Hardware mit begrenzten Ressourcen (RPi 4, 1GB RAM) optimiert und löst das Problem der Erkennung von sitzenden (statischen) Vögeln durch eine hybride Suchstrategie.

## Projektziel
Autonome Erkennung und Vertreibung von Vögeln in einem definierten Bereich. Das System unterscheidet zwischen "bewegten Zielen" (Anflug) und "statischen Zielen" (bereits gelandet).

## Hardware
* **Controller:** Raspberry Pi 4 Model B (1GB RAM)
* **Sensor:** Pi Camera Module V2 (NoIR)
* **Aktorik:** 2x Servos (Pan, Tilt)
* **Auslöser:** PIR Motion Sensor (optional, als Wake-Up)
* **Pan-Tilt-Station:**
<img width="1448" height="1195" alt="image" src="https://github.com/user-attachments/assets/fb6f073a-63f5-48ba-a3a7-94ab76889e4a" />


## Software Architektur & Logik

### 1. Vision Engine (Hybrid-Suche)
Da die NoIR-Kamera bei Tageslicht einen starken Rotstich aufweist, arbeitet die Pipeline vollständig in **Grayscale**. Die Erkennung erfolgt über TensorFlow Lite (MobileNet V2 SSD).
* **Modus A: Motion-Guided (Priorität)**
    * Erkennt Bewegung im Bild (Differenzbild).
    * Schneidet den bewegten Bereich aus (Dynamic ROI).
    * Inferenz läuft nur auf diesem Ausschnitt -> *Maximale Performance & Zoom-Effekt.*
* **Modus B: Grid-Scan (Fallback)**
    * Wenn keine Bewegung erkannt wird (Vogel sitzt still), wird das Bild in 4 Quadranten unterteilt.
    * Sequenzielle Inferenz auf den Quadranten -> *Findet schlafende/sitzende Vögel.*

### 2. High-Performance Tracking
Um die CPU zu entlasten, wird nach der initialen Erkennung **nicht** mehr per Neural Network gesucht.
* Übergabe an **OpenCV Tracker** (CSRT oder MOSSE).
* Verfolgt Textur/Muster statt Objekte -> *Funktioniert auch, wenn der Vogel stillhält.*
* **Re-Validierung:** Alle 30 Frames prüft das Neural Network kurz, ob das getrackte Objekt noch ein Vogel ist.

### 3. Ballistics & Servo Control
* PID-Regler für sanfte Kamerabewegungen.
* **Gravity Compensation:** Zielkorrektur (höher zielen) basierend auf der Entfernung (geschätzt durch Bounding-Box Größe).

