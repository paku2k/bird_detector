#!/usr/bin/env python3
"""
benchmark_detectors.py

Compares FPS performance of different YOLO11n TFLite models
on Raspberry Pi 4 with real camera input.
"""

import sys
import os
import time
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision import CameraStream
from src.yolo import YoloDetector
import config

# Test configuration
MODELS_TO_TEST = [
    {"name": "YOLO11n-320", "path": "models/yolo11n_320_int8.tflite"},
    {"name": "YOLO11n-480", "path": "models/yolo11n_480_int8.tflite"},
    {"name": "YOLO11n-640", "path": "models/yolo11n_640_int8.tflite"},
]

WARMUP_FRAMES = 10      # Frames to skip for model warmup
TEST_FRAMES = 100       # Number of frames to test per model
CONFIDENCE_THRESHOLD = 0.5

def test_model_fps(model_config, camera):
    """
    Test a single model and return FPS metrics
    """
    model_name = model_config["name"]
    model_path = os.path.join(config.BASE_DIR, model_config["path"])
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    # Temporarily change MODEL_PATH in config
    original_model_path = config.MODEL_PATH
    config.MODEL_PATH = model_path
    
    try:
        # Initialize detector
        detector = YoloDetector()
        
        # Warmup phase
        print(f"Warming up ({WARMUP_FRAMES} frames)...")
        for i in range(WARMUP_FRAMES):
            frame = camera.read()
            if frame is not None:
                _ = detector.detect(frame)
        
        # Actual test
        print(f"Running benchmark ({TEST_FRAMES} frames)...")
        
        detection_times = []
        total_detections = 0
        bird_detections = 0
        
        start_time = time.time()
        
        for i in range(TEST_FRAMES):
            frame = camera.read()
            if frame is None:
                continue
            
            # Time the detection
            det_start = time.time()
            found, score, box = detector.detect(frame)
            det_end = time.time()
            
            detection_time = (det_end - det_start) * 1000  # Convert to ms
            detection_times.append(detection_time)
            
            if found:
                total_detections += 1
                if score >= CONFIDENCE_THRESHOLD:
                    bird_detections += 1
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{TEST_FRAMES} frames")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        avg_detection_time = np.mean(detection_times)
        min_detection_time = np.min(detection_times)
        max_detection_time = np.max(detection_times)
        std_detection_time = np.std(detection_times)
        
        avg_fps = TEST_FRAMES / total_time
        theoretical_max_fps = 1000.0 / avg_detection_time  # Based on avg inference time
        
        results = {
            "model_name": model_name,
            "model_path": model_path,
            "avg_fps": avg_fps,
            "theoretical_max_fps": theoretical_max_fps,
            "avg_inference_ms": avg_detection_time,
            "min_inference_ms": min_detection_time,
            "max_inference_ms": max_detection_time,
            "std_inference_ms": std_detection_time,
            "total_detections": total_detections,
            "bird_detections": bird_detections,
            "test_frames": TEST_FRAMES,
            "total_time": total_time
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Restore original model path
        config.MODEL_PATH = original_model_path


def print_results(all_results):
    """
    Print formatted comparison table
    """
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<20} {'Avg FPS':<12} {'Max FPS':<12} {'Avg Inf (ms)':<15} {'Min/Max (ms)':<20}")
    print("-" * 80)
    
    # Sort by FPS (descending)
    sorted_results = sorted(all_results, key=lambda x: x["avg_fps"], reverse=True)
    
    for result in sorted_results:
        model_name = result["model_name"]
        avg_fps = f"{result['avg_fps']:.2f}"
        max_fps = f"{result['theoretical_max_fps']:.2f}"
        avg_inf = f"{result['avg_inference_ms']:.2f}"
        min_max = f"{result['min_inference_ms']:.1f} / {result['max_inference_ms']:.1f}"
        
        print(f"{model_name:<20} {avg_fps:<12} {max_fps:<12} {avg_inf:<15} {min_max:<20}")
    
    # Detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    for result in sorted_results:
        print(f"\n{result['model_name']}:")
        print(f"  Average FPS: {result['avg_fps']:.2f}")
        print(f"  Theoretical Max FPS: {result['theoretical_max_fps']:.2f}")
        print(f"  Average Inference Time: {result['avg_inference_ms']:.2f} ms")
        print(f"  Std Dev: {result['std_inference_ms']:.2f} ms")
        print(f"  Min Inference Time: {result['min_inference_ms']:.2f} ms")
        print(f"  Max Inference Time: {result['max_inference_ms']:.2f} ms")
        print(f"  Total Time: {result['total_time']:.2f} seconds")
        print(f"  Detections: {result['total_detections']} total, {result['bird_detections']} birds (>{CONFIDENCE_THRESHOLD*100:.0f}%)")
    
    # Winner
    print("\n" + "="*80)
    winner = sorted_results[0]
    print(f"🏆 WINNER: {winner['model_name']} with {winner['avg_fps']:.2f} FPS")
    print("="*80)


def main():
    print("🚀 YOLO11n Model FPS Benchmark")
    print(f"Camera Resolution: {config.CAMERA_RES}")
    print(f"Camera FPS: {config.CAMERA_FPS}")
    print(f"Test Frames per Model: {TEST_FRAMES}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    
    # Initialize camera
    print("\n📷 Initializing camera...")
    camera = CameraStream(resolution=config.CAMERA_RES, fps=config.CAMERA_FPS)
    camera.start()
    time.sleep(2.0)  # Let camera stabilize
    print("✅ Camera ready")
    
    # Test each model
    all_results = []
    
    for model_config in MODELS_TO_TEST:
        result = test_model_fps(model_config, camera)
        if result is not None:
            all_results.append(result)
        
        # Small delay between models
        time.sleep(1.0)
    
    # Stop camera
    camera.stop()
    
    # Print results
    if all_results:
        print_results(all_results)
        
        # Save results to file
        results_file = "benchmark_results.txt"
        with open(results_file, 'w') as f:
            f.write("YOLO11n Model FPS Benchmark Results\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Camera: {config.CAMERA_RES} @ {config.CAMERA_FPS} FPS\n")
            f.write(f"Test Frames: {TEST_FRAMES}\n\n")
            
            for result in sorted(all_results, key=lambda x: x["avg_fps"], reverse=True):
                f.write(f"\n{result['model_name']}:\n")
                f.write(f"  Average FPS: {result['avg_fps']:.2f}\n")
                f.write(f"  Avg Inference: {result['avg_inference_ms']:.2f} ms\n")
                f.write(f"  Min/Max Inference: {result['min_inference_ms']:.1f} / {result['max_inference_ms']:.1f} ms\n")
                f.write(f"  Detections: {result['total_detections']} total, {result['bird_detections']} birds\n")
        
        print(f"\n💾 Results saved to: {results_file}")
    else:
        print("\n❌ No successful tests completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
