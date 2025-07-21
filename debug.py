#!/usr/bin/env python3
"""
Safe Web Streamer - DPU 호출 완전 비활성화 버전
segfault 원인을 찾기 위한 테스트 버전
"""

import cv2
import numpy as np
import time
import json
import os
import threading
from flask import Flask, Response, render_template_string

class SafeWebStreamer:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        # Flask app
        self.app = Flask(__name__)
        
        # Communication files (하지만 사용하지 않음)
        self.input_file = "/tmp/dpu_input.jpg"
        self.result_file = "/tmp/dpu_result.json"
        self.status_file = "/tmp/dpu_status.txt"
        
        # Camera
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Fake DPU results (실제 DPU 호출 안 함)
        self.fake_detections = 0
        self.fake_inference_time = 0.1
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>🧪 Safe Web Streamer (No DPU)</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #1a1a1a;
            color: white;
            text-align: center;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .stats { 
            display: flex; 
            justify-content: space-around; 
            margin: 20px 0;
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
        }
        .stat-item { 
            text-align: center;
        }
        .stat-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: #ff6b6b;
        }
        .stat-label { 
            font-size: 14px; 
            color: #ccc;
        }
        img { 
            max-width: 100%; 
            border: 2px solid #ff6b6b;
            border-radius: 10px;
        }
        .test-mode {
            background: rgba(255,107,107,0.2);
            padding: 10px;
            border-radius: 8px;
            margin: 20px 0;
            color: #ff6b6b;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Safe Web Streamer</h1>
        
        <div class="test-mode">
            ⚠️ TEST MODE: DPU calls completely disabled to isolate segfault cause
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="fps">--</div>
                <div class="stat-label">Camera FPS</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="detections">FAKE</div>
                <div class="stat-label">Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="inference-time">N/A</div>
                <div class="stat-label">DPU Time (ms)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="status">🔴</div>
                <div class="stat-label">DPU Status</div>
            </div>
        </div>
        
        <img src="/video_feed" alt="Camera Stream" />
        
        <div class="test-mode">
            If this runs without segfault, the issue is in DPU worker process
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('detections').textContent = 'FAKE';
                    document.getElementById('inference-time').textContent = 'N/A';
                    document.getElementById('status').textContent = '🔴';
                })
                .catch(error => console.error('Error:', error));
        }
        
        setInterval(updateStats, 1000);
        updateStats();
    </script>
</body>
</html>
            """)
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def stats():
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            return {
                'fps': fps,
                'detections': 'DISABLED',
                'inference_time': 0,
                'dpu_status': False
            }
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.start_time = time.time()
        
        print("Camera started (SAFE MODE - NO DPU)")
        return True
    
    def generate_frames(self):
        """Generate frames WITHOUT any DPU communication"""
        while True:
            try:
                if self.is_running and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_count += 1
                        current_time = time.time()
                        
                        # NO DPU CALLS AT ALL - just display camera
                        display_frame = frame.copy()
                        
                        # Add overlays
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        
                        cv2.putText(display_frame, f"SAFE MODE - FPS: {fps:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(display_frame, "DPU: DISABLED", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Frame: {self.frame_count}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, "NO SEGFAULT TEST", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Encode frame
                        _, buffer = cv2.imencode('.jpg', display_frame, 
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        time.sleep(0.1)
                else:
                    # Default frame
                    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(default_frame, "SAFE MODE - Camera Not Started", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', default_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Safe frame generation error: {e}")
                # Create error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "SAFE MODE ERROR", 
                           (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(error_frame, str(e)[:30], 
                           (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        if not self.start_camera():
            return
        
        print(f"🧪 SAFE web streamer: http://{host}:{port}")
        print("🔴 DPU calls completely DISABLED")
        print("📹 Camera-only mode for segfault isolation")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\nSafe web streamer stopping...")
        finally:
            self.is_running = False
            if self.cap:
                self.cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    
    streamer = SafeWebStreamer(args.camera)
    streamer.run(host=args.host, port=args.port)
