#!/usr/bin/env python3
"""
Single-threaded Web Detection - No Flask threading for DPU calls
"""

import cv2
import numpy as np
import time
import gc
import argparse
from flask import Flask, Response, render_template_string
import queue
import threading

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class SingleThreadDetector:
    def __init__(self, model_path: str, classes_file: str):
        self.model_path = model_path
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Load model
        self._load_model()
        
        # Single detection result storage
        self.latest_result = None
        self.latest_frame = None
        self.detection_lock = threading.Lock()
        
        # Stats
        self.inference_time = 0
        self.detection_count = 0
        
    def _load_model(self):
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs with full XIR API compatibility
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    children = root_subgraph.children_topological_sort()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)
                    else:
                        subgraphs = [children] if children else []
                elif hasattr(root_subgraph, 'get_children'):
                    children = root_subgraph.get_children()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)
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
            
            print(f"Found {len(subgraphs)} subgraphs:")
            
            # Debug: Print ALL subgraphs
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
            
            # Method 2: Try indices that worked in previous tests (1, 2)
            if dpu_subgraph is None:
                print("Method 1 failed, trying known working indices...")
                for idx in [1, 2, 0]:  # Try indices from successful tests
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
            
            if dpu_subgraph is None:
                print("\n❌ COMPLETE FAILURE - No working subgraph found")
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
                raise ValueError("No DPU subgraph found")
            
            # Create final DPU runner
            print(f"Creating final DPU runner with: {dpu_subgraph.get_name()}")
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            if self.dpu_runner is None:
                raise ValueError("Final DPU runner creation failed")
            
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            if not self.input_tensors or not self.output_tensors:
                raise ValueError("Failed to get tensor information")
            
            print("✅ DPU model loaded successfully")
            print(f"Final subgraph: {dpu_subgraph.get_name()}")
            print(f"Input: {self.input_tensors[0].dims}")
            print(f"Outputs: {[t.dims for t in self.output_tensors]}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    def process_single_frame(self, frame):
        """Process ONE frame using the PROVEN method from final_dpu_test.py"""
        try:
            start_time = time.time()
            
            # Preprocess exactly like Method 2 from final_dpu_test.py
            img_h, img_w = frame.shape[:2]
            scale = min(416 / img_w, 416 / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            padded = np.full((416, 416, 3), 114, dtype=np.uint8)
            pad_x = (416 - new_w) // 2
            pad_y = (416 - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # CRITICAL: Use EXACT method from final_dpu_test.py Method 2
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)  # CRITICAL!
            
            # Manual output allocation (EXACT Method 2)
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)  # CRITICAL!
                output_arrays.append(output_array)
            
            # Clear memory (from Method 2)
            gc.collect()
            
            # EXACT DPU call from Method 2
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            # Simple detection count
            detection_count = 0
            for output in output_arrays:
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    objectness = output[0, :, :, 4]
                    detection_count += np.sum(objectness > 0.3)
            
            self.inference_time = time.time() - start_time
            self.detection_count = min(detection_count, 50)  # Cap at 50
            
            # Create result frame
            result_frame = frame.copy()
            cv2.putText(result_frame, f"Detections: {self.detection_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Inference: {self.inference_time*1000:.1f}ms", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, "SINGLE THREAD", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            return result_frame, True
            
        except Exception as e:
            print(f"Detection error: {e}")
            # Return original frame on error
            error_frame = frame.copy()
            cv2.putText(error_frame, f"ERROR: {str(e)[:30]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return error_frame, False

class SingleThreadWebApp:
    def __init__(self, detector, camera_id=0):
        self.detector = detector
        self.camera_id = camera_id
        
        # Flask app with threading disabled for DPU calls
        self.app = Flask(__name__)
        self.app.config['THREADED'] = False  # Disable threading
        
        # Single frame storage
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Camera
        self.cap = None
        self.is_running = False
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🔧 Single-thread Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #1a1a1a;
            color: white;
            text-align: center;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
        }
        img { 
            max-width: 100%; 
            border: 2px solid #00ff00;
            border-radius: 10px;
        }
        .status {
            margin: 20px 0;
            font-size: 18px;
            color: #00ff00;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Single-thread DPU Detection</h1>
        <div class="status">No Flask threading - Direct DPU calls</div>
        <img src="/video_feed" alt="Detection Stream" />
        <div class="status">Method 2 from final_dpu_test.py</div>
    </div>
</body>
</html>
            ''')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.start_time = time.time()
        print("Camera started (single-thread mode)")
        return True
    
    def generate_frames(self):
        """Generate frames with SAFE error handling"""
        frame_counter = 0
        
        while True:
            try:
                if self.is_running and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        frame_counter += 1
                        
                        # Start with simple frame display, add detection gradually
                        if frame_counter < 10:
                            # First 10 frames: no detection, just display
                            display_frame = frame.copy()
                            cv2.putText(display_frame, "Initializing...", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            cv2.putText(display_frame, f"Frame: {frame_counter}", 
                                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                        elif frame_counter < 20:
                            # Next 10 frames: test detection but don't fail if error
                            display_frame = frame.copy()
                            cv2.putText(display_frame, "Testing detection...", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            
                            # Try detection but catch ALL errors
                            try:
                                if frame_counter % 5 == 0:  # Test every 5th frame
                                    print(f"Testing detection on frame {frame_counter}")
                                    result_frame, success = self.detector.process_single_frame(frame)
                                    if success:
                                        display_frame = result_frame
                                        cv2.putText(display_frame, "Detection OK!", 
                                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    else:
                                        cv2.putText(display_frame, "Detection Failed", 
                                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            except Exception as e:
                                print(f"Detection test failed: {e}")
                                cv2.putText(display_frame, f"Error: {str(e)[:20]}", 
                                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        else:
                            # After frame 20: normal operation
                            self.frame_count += 1
                            
                            # Process every 3rd frame to reduce load
                            if self.frame_count % 3 == 0:
                                try:
                                    # SINGLE-THREADED detection call with full error protection
                                    result_frame, success = self.detector.process_single_frame(frame)
                                    
                                    if success:
                                        # Add FPS info
                                        elapsed = time.time() - self.start_time
                                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                                        cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                        
                                        with self.frame_lock:
                                            self.current_frame = result_frame
                                    else:
                                        # Detection failed, use original frame
                                        error_frame = frame.copy()
                                        cv2.putText(error_frame, "Detection Error", 
                                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                        with self.frame_lock:
                                            self.current_frame = error_frame
                                            
                                except Exception as e:
                                    print(f"Critical detection error: {e}")
                                    # Even if detection completely fails, show original frame
                                    error_frame = frame.copy()
                                    cv2.putText(error_frame, "CRITICAL ERROR", 
                                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.putText(error_frame, str(e)[:30], 
                                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    with self.frame_lock:
                                        self.current_frame = error_frame
                            
                            # Use current processed frame
                            with self.frame_lock:
                                if self.current_frame is not None:
                                    display_frame = self.current_frame
                                else:
                                    display_frame = frame
                        
                        # Encode and yield frame
                        try:
                            _, buffer = cv2.imencode('.jpg', display_frame, 
                                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        except Exception as e:
                            print(f"Frame encoding error: {e}")
                            # Create emergency frame
                            emergency_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(emergency_frame, "ENCODING ERROR", (200, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            _, buffer = cv2.imencode('.jpg', emergency_frame)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        time.sleep(0.1)
                else:
                    # Camera not running - show default frame
                    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(default_frame, "Camera Not Started", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(default_frame, "Single Thread Mode", (200, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    _, buffer = cv2.imencode('.jpg', default_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS limit
                
            except Exception as e:
                print(f"Generate frames critical error: {e}")
                # Emergency response - create error frame
                try:
                    emergency_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(emergency_frame, "STREAM ERROR", (200, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(emergency_frame, str(e)[:40], (50, 260), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(emergency_frame, "Continuing...", (200, 300), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    _, buffer = cv2.imencode('.jpg', emergency_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(1)  # Wait before continuing
                except:
                    # If even emergency frame fails, just wait
                    time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        if not self.start_camera():
            return
        
        print(f"🔧 Single-thread web server: http://{host}:{port}")
        print("⚠️  DPU calls are NOT threaded - should prevent segfault")
        
        try:
            # Run Flask in single-threaded mode
            self.app.run(host=host, port=port, debug=False, threaded=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.is_running = False
            if self.cap:
                self.cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    parser.add_argument('--classes', default='coco2017_classes.txt')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🔧 Initializing single-thread detector...")
    detector = SingleThreadDetector(args.model, args.classes)
    
    print("🌐 Starting single-thread web app...")
    app = SingleThreadWebApp(detector, args.camera)
    
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
