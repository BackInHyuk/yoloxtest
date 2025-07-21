#!/usr/bin/env python3
"""
Separated Process Solution
1. dpu_worker.py - DPU 전용 프로세스
2. web_streamer.py - 웹 스트리밍 전용 프로세스
3. 파일 기반 통신
"""

# =============================================================================
# 파일 1: dpu_worker.py (DPU 전용 워커)
# =============================================================================

DPU_WORKER_CODE = '''#!/usr/bin/env python3
"""
DPU Worker Process - DPU 전용 프로세스
웹서버와 완전 분리되어 안전하게 실행
"""

import time
import numpy as np
import gc
import json
import os
import sys

try:
    import vart
    import xir
    import cv2
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class DPUWorker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Communication files
        self.input_file = "/tmp/dpu_input.jpg"
        self.result_file = "/tmp/dpu_result.json"
        self.status_file = "/tmp/dpu_status.txt"
        
        # Load model
        self._load_model()
        
        # Write status
        with open(self.status_file, 'w') as f:
            f.write("DPU_READY")
        
        print("DPU Worker ready")
    
    def _load_model(self):
        """Load DPU model with COMPLETE error handling from all previous attempts"""
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs with FULL XIR API compatibility (ALL previous errors fixed)
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    children = root_subgraph.children_topological_sort()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)  # Fix 'set' object is not subscriptable
                    else:
                        subgraphs = [children] if children else []
                elif hasattr(root_subgraph, 'get_children'):
                    children = root_subgraph.get_children()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)  # Fix set issue
                    else:
                        subgraphs = [children] if children else []
                else:
                    print("Using root subgraph directly")
                    subgraphs = [root_subgraph]
            except Exception as e:
                print(f"Error getting subgraphs: {e}")
                subgraphs = [root_subgraph]
            
            if not subgraphs:
                raise ValueError("No subgraphs found")
            
            print(f"DPU Worker: Found {len(subgraphs)} subgraphs:")
            
            # Debug: Print ALL subgraphs (from previous successful attempts)
            for i, sg in enumerate(subgraphs):
                if sg is None:
                    print(f"  Subgraph {i}: None")
                    continue
                try:
                    name = sg.get_name()
                    device = "Unknown"
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                    print(f"  Subgraph {i}: {name} (device: {device})")
                except Exception as e:
                    print(f"  Subgraph {i}: Error - {e}")
            
            # Method 1: Find by DPU device attribute
            dpu_subgraph = None
            for i, sg in enumerate(subgraphs):
                if sg is None:
                    continue
                try:
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = sg
                            print(f"✅ Method 1 SUCCESS: Found DPU at index {i}")
                            break
                except Exception as e:
                    print(f"  Method 1 error at index {i}: {e}")
                    continue
            
            # Method 2: Try indices that worked in previous tests (1, 2, 0, 3, 4)
            if dpu_subgraph is None:
                print("Method 1 failed, trying known working indices...")
                for idx in [1, 2, 0, 3, 4, 5]:  # Extended range
                    if len(subgraphs) > idx and subgraphs[idx] is not None:
                        try:
                            sg = subgraphs[idx]
                            name = sg.get_name()
                            print(f"  Testing subgraph {idx}: {name}")
                            
                            if name != "root":
                                # Try to create a test runner
                                test_runner = vart.Runner.create_runner(sg, "run")
                                if test_runner is not None:
                                    dpu_subgraph = sg
                                    print(f"✅ Method 2 SUCCESS: Validated subgraph {idx}")
                                    break
                                else:
                                    print(f"  Subgraph {idx}: Runner creation returned None")
                            else:
                                print(f"  Subgraph {idx}: Skipped (root)")
                        except Exception as e:
                            print(f"  Subgraph {idx} failed: {e}")
                            continue
            
            # Method 3: Try ANY non-root subgraph
            if dpu_subgraph is None:
                print("Method 2 failed, trying any non-root subgraph...")
                for i, sg in enumerate(subgraphs):
                    if sg is None:
                        continue
                    try:
                        name = sg.get_name()
                        print(f"  Testing any subgraph {i}: {name}")
                        
                        if name != "root":
                            try:
                                test_runner = vart.Runner.create_runner(sg, "run")
                                if test_runner is not None:
                                    dpu_subgraph = sg
                                    print(f"✅ Method 3 SUCCESS: Found working subgraph {i}")
                                    break
                            except Exception as e:
                                print(f"    Failed: {e}")
                                continue
                    except Exception as e:
                        print(f"  Error with subgraph {i}: {e}")
                        continue
            
            # Method 4: Last resort - try even root if necessary
            if dpu_subgraph is None:
                print("All methods failed, trying root subgraph as last resort...")
                try:
                    test_runner = vart.Runner.create_runner(root_subgraph, "run")
                    if test_runner is not None:
                        dpu_subgraph = root_subgraph
                        print("✅ Method 4 SUCCESS: Using root subgraph")
                except Exception as e:
                    print(f"Root subgraph also failed: {e}")
            
            # Method 5: Try alternative subgraph access methods
            if dpu_subgraph is None:
                print("Trying alternative subgraph access methods...")
                try:
                    # Try getting subgraphs differently
                    all_subgraphs = []
                    
                    # Method A: Direct iteration if possible
                    if hasattr(root_subgraph, '__iter__'):
                        try:
                            for sg in root_subgraph:
                                all_subgraphs.append(sg)
                        except:
                            pass
                    
                    # Method B: Try get_subgraph if exists
                    if hasattr(graph, 'get_subgraph'):
                        try:
                            for i in range(10):  # Try first 10 indices
                                try:
                                    sg = graph.get_subgraph(i)
                                    if sg is not None:
                                        all_subgraphs.append(sg)
                                except:
                                    break
                        except:
                            pass
                    
                    print(f"Alternative method found {len(all_subgraphs)} additional subgraphs")
                    
                    # Test these alternative subgraphs
                    for i, sg in enumerate(all_subgraphs):
                        if sg is None:
                            continue
                        try:
                            test_runner = vart.Runner.create_runner(sg, "run")
                            if test_runner is not None:
                                dpu_subgraph = sg
                                print(f"✅ Method 5 SUCCESS: Alternative subgraph {i}")
                                break
                        except:
                            continue
                            
                except Exception as e:
                    print(f"Alternative methods failed: {e}")
            
            # Method 6: Desperate attempt - try first available subgraph
            if dpu_subgraph is None:
                print("Desperate attempt: trying first available subgraph...")
                for sg in subgraphs:
                    if sg is not None:
                        try:
                            dpu_subgraph = sg
                            print("Using first available subgraph (desperate)")
                            break
                        except:
                            continue
            
            if dpu_subgraph is None:
                print("\\n❌ COMPLETE FAILURE - No working subgraph found")
                print("Available subgraphs:")
                for i, sg in enumerate(subgraphs):
                    if sg is not None:
                        try:
                            name = sg.get_name()
                            device = "Unknown"
                            if sg.has_attr("device"):
                                device = sg.get_attr("device")
                            print(f"  {i}: {name} (device: {device})")
                        except:
                            print(f"  {i}: Error getting info")
                raise ValueError("No DPU subgraph found after all 6 methods")
            
            # Create final DPU runner
            print(f"Creating final DPU runner with: {dpu_subgraph.get_name()}")
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            if self.dpu_runner is None:
                raise ValueError("Final DPU runner creation failed")
            
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            if not self.input_tensors or not self.output_tensors:
                raise ValueError("Failed to get tensor information")
            
            print("✅ DPU model loaded successfully in worker process")
            print(f"Final subgraph: {dpu_subgraph.get_name()}")
            print(f"Input: {self.input_tensors[0].dims}")
            print(f"Outputs: {[t.dims for t in self.output_tensors]}")
            
        except Exception as e:
            print(f"DPU worker model loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def process_image(self, image_path):
        """Process single image with proven method"""
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                return {"success": False, "error": "Could not read image"}
            
            start_time = time.time()
            
            # Preprocess (proven Method 2)
            img_h, img_w = frame.shape[:2]
            scale = min(416 / img_w, 416 / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            padded = np.full((416, 416, 3), 114, dtype=np.uint8)
            pad_x = (416 - new_w) // 2
            pad_y = (416 - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # Convert to int8 (proven method)
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)
            
            # Prepare outputs
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # Clear memory
            gc.collect()
            
            # DPU inference (proven method)
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            # Count detections
            detection_count = 0
            for output in output_arrays:
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    objectness = output[0, :, :, 4]
                    detection_count += np.sum(objectness > 0.3)
            
            inference_time = time.time() - start_time
            
            result = {
                "success": True,
                "detections": min(int(detection_count), 50),
                "inference_time": float(inference_time),
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def run(self):
        """Main worker loop"""
        print("DPU Worker started - waiting for requests...")
        
        last_check = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Check for new input every 0.5 seconds
                if current_time - last_check > 0.5:
                    if os.path.exists(self.input_file):
                        print("Processing new image...")
                        
                        # Process the image
                        result = self.process_image(self.input_file)
                        
                        # Write result
                        with open(self.result_file, 'w') as f:
                            json.dump(result, f)
                        
                        # Remove input file to signal completion
                        try:
                            os.remove(self.input_file)
                        except:
                            pass
                        
                        if result["success"]:
                            print(f"Success: {result['detections']} detections in {result['inference_time']:.3f}s")
                        else:
                            print(f"Failed: {result['error']}")
                    
                    last_check = current_time
                
                time.sleep(0.1)  # Prevent CPU overload
                
            except KeyboardInterrupt:
                print("DPU Worker stopping...")
                break
            except Exception as e:
                print(f"DPU Worker error: {e}")
                time.sleep(1)
        
        # Cleanup
        for f in [self.result_file, self.status_file]:
            try:
                os.remove(f)
            except:
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    args = parser.parse_args()
    
    worker = DPUWorker(args.model)
    worker.run()
'''

# =============================================================================
# 파일 2: web_streamer.py (웹 스트리밍 전용)
# =============================================================================

WEB_STREAMER_CODE = '''#!/usr/bin/env python3
"""
Web Streamer Process - 웹 스트리밍 전용 프로세스
DPU와 완전 분리되어 안전하게 실행
"""

import cv2
import time
import json
import os
import threading
from flask import Flask, Response, render_template_string

class WebStreamer:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        # Flask app
        self.app = Flask(__name__)
        
        # Communication files
        self.input_file = "/tmp/dpu_input.jpg"
        self.result_file = "/tmp/dpu_result.json"
        self.status_file = "/tmp/dpu_status.txt"
        
        # Camera
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # DPU results
        self.last_result = {"detections": 0, "inference_time": 0}
        self.last_detection_time = 0
        
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
    <title>🔄 Separated Process Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #0f1419;
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
            color: #00d4aa;
        }
        .stat-label { 
            font-size: 14px; 
            color: #ccc;
        }
        img { 
            max-width: 100%; 
            border: 2px solid #00d4aa;
            border-radius: 10px;
        }
        .process-info {
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Separated Process Detection</h1>
        
        <div class="process-info">
            <strong>Architecture:</strong> Web Process ↔️ File Communication ↔️ DPU Process
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="fps">--</div>
                <div class="stat-label">Web FPS</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="detections">--</div>
                <div class="stat-label">Detections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="inference-time">--</div>
                <div class="stat-label">DPU Time (ms)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="status">🔄</div>
                <div class="stat-label">Status</div>
            </div>
        </div>
        
        <img src="/video_feed" alt="Camera Stream" />
        
        <div class="process-info">
            ✅ No segfaults - Processes are completely separated
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('inference-time').textContent = (data.inference_time * 1000).toFixed(1);
                    document.getElementById('status').textContent = data.dpu_status ? '🟢' : '🔴';
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
            
            # Check DPU status
            dpu_status = os.path.exists(self.status_file)
            
            return {
                'fps': fps,
                'detections': self.last_result.get('detections', 0),
                'inference_time': self.last_result.get('inference_time', 0),
                'dpu_status': dpu_status
            }
    
    def check_dpu_status(self):
        """Check if DPU worker is ready"""
        return os.path.exists(self.status_file)
    
    def request_detection(self, frame):
        """Request detection from DPU worker"""
        try:
            # Save frame for DPU worker
            success = cv2.imwrite(self.input_file, frame)
            if success:
                return True
        except Exception as e:
            print(f"Error requesting detection: {e}")
        return False
    
    def get_detection_result(self):
        """Get result from DPU worker"""
        try:
            if os.path.exists(self.result_file):
                with open(self.result_file, 'r') as f:
                    result = json.load(f)
                
                # Remove result file
                try:
                    os.remove(self.result_file)
                except:
                    pass
                
                return result
        except Exception as e:
            print(f"Error getting result: {e}")
        
        return None
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.start_time = time.time()
        
        print("Camera started (web process)")
        return True
    
    def generate_frames(self):
        """Generate frames with separated DPU communication"""
        detection_pending = False
        
        while True:
            try:
                if self.is_running and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame_count += 1
                        current_time = time.time()
                        
                        # Check for detection result
                        if detection_pending:
                            result = self.get_detection_result()
                            if result is not None:
                                self.last_result = result
                                detection_pending = False
                                if result["success"]:
                                    print(f"Got DPU result: {result['detections']} detections")
                                else:
                                    print(f"DPU error: {result['error']}")
                        
                        # Request new detection every 3 seconds
                        if (not detection_pending and 
                            current_time - self.last_detection_time > 3.0 and
                            self.check_dpu_status()):
                            
                            if self.request_detection(frame):
                                detection_pending = True
                                self.last_detection_time = current_time
                                print("Requested DPU detection")
                        
                        # Add overlay
                        display_frame = frame.copy()
                        
                        # FPS
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        cv2.putText(display_frame, f"Web FPS: {fps:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # DPU status
                        if self.check_dpu_status():
                            cv2.putText(display_frame, "DPU: ONLINE", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(display_frame, "DPU: OFFLINE", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Detection info
                        if detection_pending:
                            cv2.putText(display_frame, "DPU Processing...", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        else:
                            cv2.putText(display_frame, f"Detections: {self.last_result.get('detections', 0)}", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        cv2.putText(display_frame, "SEPARATED PROCESSES", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        
                        # Encode frame
                        _, buffer = cv2.imencode('.jpg', display_frame, 
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        yield (b'--frame\\r\\n'
                               b'Content-Type: image/jpeg\\r\\n\\r\\n' + buffer.tobytes() + b'\\r\\n')
                    else:
                        time.sleep(0.1)
                else:
                    # Default frame
                    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(default_frame, "Camera Not Started", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', default_frame)
                    yield (b'--frame\\r\\n'
                           b'Content-Type: image/jpeg\\r\\n\\r\\n' + buffer.tobytes() + b'\\r\\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Frame generation error: {e}")
                time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        if not self.start_camera():
            return
        
        print(f"🌐 Web streamer: http://{host}:{port}")
        print("🔄 Separated from DPU process")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\\nWeb streamer stopping...")
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
    
    streamer = WebStreamer(args.camera)
    streamer.run(host=args.host, port=args.port)
'''

# =============================================================================
# 파일 3: run_separated.py (통합 실행기)
# =============================================================================

RUN_SEPARATED_CODE = '''#!/usr/bin/env python3
"""
Separated Process Runner
DPU와 웹서버를 별도 프로세스로 실행
"""

import subprocess
import time
import sys
import signal
import os

def cleanup():
    """Cleanup function"""
    print("\\nCleaning up...")
    # Kill any remaining processes
    try:
        subprocess.run(["pkill", "-f", "dpu_worker.py"], check=False)
        subprocess.run(["pkill", "-f", "web_streamer.py"], check=False)
    except:
        pass
    
    # Remove temp files
    temp_files = ["/tmp/dpu_input.jpg", "/tmp/dpu_result.json", "/tmp/dpu_status.txt"]
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🔄 Starting Separated Process Detection System")
    print("=" * 60)
    
    # Start DPU worker
    print("1. Starting DPU worker process...")
    dpu_process = subprocess.Popen([
        sys.executable, "dpu_worker.py", 
        "--model", "yolox_nano_pt.xmodel"
    ])
    
    # Wait for DPU to be ready
    print("2. Waiting for DPU worker to be ready...")
    timeout = 30
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if os.path.exists("/tmp/dpu_status.txt"):
            print("✅ DPU worker is ready!")
            break
        time.sleep(1)
    else:
        print("❌ DPU worker failed to start within timeout")
        cleanup()
        return
    
    # Start web streamer
    print("3. Starting web streamer process...")
    web_process = subprocess.Popen([
        sys.executable, "web_streamer.py",
        "--camera", "0",
        "--host", "0.0.0.0", 
        "--port", "5000"
    ])
    
    print("\\n🚀 System started successfully!")
    print("📱 Web interface: http://YOUR_IP:5000")
    print("🔄 DPU and Web processes are completely separated")
    print("\\nPress Ctrl+C to stop all processes")
    
    try:
        # Monitor processes
        while True:
            if dpu_process.poll() is not None:
                print("❌ DPU worker process died")
                break
            if web_process.poll() is not None:
                print("❌ Web streamer process died")
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\\nStopping processes...")
        try:
            dpu_process.terminate()
            web_process.terminate()
            
            # Wait for graceful shutdown
            try:
                dpu_process.wait(timeout=5)
                web_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing processes...")
                dpu_process.kill()
                web_process.kill()
        except:
            pass
        
        cleanup()
        print("✅ All processes stopped")

if __name__ == "__main__":
    main()
'''

def create_separated_files():
    """Create the separated process files"""
    
    # Create dpu_worker.py
    with open("dpu_worker.py", "w") as f:
        f.write(DPU_WORKER_CODE)
    
    # Create web_streamer.py  
    with open("web_streamer.py", "w") as f:
        f.write(WEB_STREAMER_CODE)
    
    # Create run_separated.py
    with open("run_separated.py", "w") as f:
        f.write(RUN_SEPARATED_CODE)
    
    # Make files executable
    import stat
    for filename in ["dpu_worker.py", "web_streamer.py", "run_separated.py"]:
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)
    
    print("✅ Created separated process files:")
    print("  - dpu_worker.py")
    print("  - web_streamer.py") 
    print("  - run_separated.py")
    print()
    print("🚀 Run with: python3 run_separated.py")

if __name__ == "__main__":
    create_separated_files()
