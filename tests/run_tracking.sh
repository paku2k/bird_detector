#!/bin/bash
# Quick launcher for advanced tracking test

echo "🎯 Advanced Bird Tracking Test Launcher"
echo "========================================"
echo ""
echo "Select configuration:"
echo "1) Fast Mode (MOSSE, no history, 30+ FPS)"
echo "2) Accurate Mode (CSRT, with history, 20+ FPS)"
echo "3) Balanced Mode (MOSSE, with history, 25+ FPS)"
echo "4) Custom (edit test_advanced_tracking.py)"
echo ""
read -p "Choice [1-4]: " choice

case $choice in
    1)
        echo "Starting Fast Mode..."
        sed -i 's/TRACKER_TYPE = .*/TRACKER_TYPE = "MOSSE"/' test_advanced_tracking.py
        sed -i 's/HISTORY_TRACK = .*/HISTORY_TRACK = 0/' test_advanced_tracking.py
        sed -i 's/YOLO_INTERVAL = .*/YOLO_INTERVAL = 3.0/' test_advanced_tracking.py
        ;;
    2)
        echo "Starting Accurate Mode..."
        sed -i 's/TRACKER_TYPE = .*/TRACKER_TYPE = "CSRT"/' test_advanced_tracking.py
        sed -i 's/HISTORY_TRACK = .*/HISTORY_TRACK = 1/' test_advanced_tracking.py
        sed -i 's/YOLO_INTERVAL = .*/YOLO_INTERVAL = 2.0/' test_advanced_tracking.py
        ;;
    3)
        echo "Starting Balanced Mode..."
        sed -i 's/TRACKER_TYPE = .*/TRACKER_TYPE = "MOSSE"/' test_advanced_tracking.py
        sed -i 's/HISTORY_TRACK = .*/HISTORY_TRACK = 1/' test_advanced_tracking.py
        sed -i 's/YOLO_INTERVAL = .*/YOLO_INTERVAL = 2.5/' test_advanced_tracking.py
        ;;
    4)
        echo "Using custom settings from test_advanced_tracking.py"
        ;;
    *)
        echo "Invalid choice, using default settings"
        ;;
esac

echo ""
echo "Starting tracker..."
echo "Web interface will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 test_advanced_tracking.py
