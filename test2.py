#!/usr/bin/env python3
"""
Completely separated camera capture and DPU processing
Use file-based communication to avoid memory conflicts
"""

import cv2
import numpy as np
import time
import gc
import argparse
import os
import threading
from flask import Flask, Response, render_template_string

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class SeparatedDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Load model
        self._load_model()
        
        # Stats
        self.inference_time = 0
        self.detection_count = 0
        
    def _load_model(self):
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs with FULL XIR API compatibility
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    children = root_subgraph.children_topological_sort()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)  # Convert set to list
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
            
            # Method 2: Try indices that worked in previous tests (1, 2, 0)
            if dpu_subgraph is None:
                print("Method 1 failed, trying known working indices...")
                for idx in [1, 2, 0, 3, 4]:  # Try more indices
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
                
                # Try one more desperate attempt with first available subgraph
                print("Desperate attempt: trying first available subgraph...")
                for sg in subgraphs:
                    if sg is not None:
                        try:
                            dpu_subgraph = sg
                            print("Using first available subgraph")
                            break
                        except:
                            continue
                
                if dpu_subgraph is None:
                    raise ValueError("No DPU subgraph found after all attempts")
            
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
    
    def process_image_file(self, image_file: str) -> tuple:
        """Process image from file (avoiding camera memory issues)"""
        try:
            start_time = time.time()
            
            # Read image from file
            frame = cv2.imread(image_file)
            if frame is None:
                return None, False, "Could not read image file"
            
            # Preprocess EXACTLY like final_dpu_test.py Method 2
            img_h, img_w = frame.shape[:2]
            scale = min(416 / img_w, 416 / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            padded = np.full((416, 416, 3), 114, dtype=np.uint8)
            pad_x = (416 - new_w) // 2
            pad_y = (416 - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # EXACT Method 2 conversion
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)
            
            # EXACT Method 2 output allocation
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # Clear memory
            gc.collect()
            
            # EXACT Method 2 DPU call
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            # Simple detection count
            detection_count = 0
            for output in output_arrays:
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    objectness = output[0, :, :, 4]
                    detection_count += np.sum(objectness > 0.3)
            
            self.inference_time = time.time() - start_time
            self.detection_count = min(detection_count, 50)
            
            # Create result frame
            result_frame = frame.copy()
            cv2.putText(result_frame, f"Detections: {self.detection_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Inference: {self.inference_time*1000:.1f}ms", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, "FILE-BASED", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            return result_frame, True, "Success"
            
        except Exception as e:
            error_msg = f"Detection error: {str(e)[:50]}"
            print(error_msg)
            
            # Return error frame if possible
            try:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "DETECTION ERROR", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(error_frame, error_msg[:30], 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return error_frame, False, error_msg
            except:
                return None, False, error_msg

class SeparatedWebApp:
    def __init__(self, detector, camera_id=0):
        self.detector = detector
        self.camera_id = camera_id
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['THREADED'] = False
        
        # File-based communication
        self.temp_image_file = "/tmp/current_frame.jpg"
        self.processed_image_file = "/tmp/processed_frame.jpg"
        
        # Camera capture thread (completely separate)
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        
        # Current frame for display
        self.current_display_frame = None
        self.frame_lock = threading.Lock()
        
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
    <title>🔄 Separated Camera+DPU</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #2a2a2a;
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
            border: 2px solid #ff6600;
            border-radius: 10px;
        }
        .status {
            margin: 20px 0;
            font-size: 18px;
            color: #ff6600;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Separated Camera + DPU Detection</h1>
        <div class="status">Camera → File → DPU (Memory Isolation)</div>
        <img src="/video_feed" alt="Detection Stream" />
        <div class="status">Avoiding direct camera-DPU memory conflicts</div>
    </div>
</body>
</html>
            ''')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def camera_capture_thread(self):
        """Completely separate camera capture thread"""
        print("Starting camera capture thread...")
        
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Save frame to file (isolation from DPU)
                        cv2.imwrite(self.temp_image_file, frame)
                        self.frame_count += 1
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.1)
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"Camera capture error: {e}")
                time.sleep(1)
        
        print("Camera capture thread stopped")
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.start_time = time.time()
        
        # Start separate capture thread
        self.capture_thread = threading.Thread(target=self.camera_capture_thread, daemon=True)
        self.capture_thread.start()
        
        print("Camera started (separated mode)")
        return True
    
    def generate_frames(self):
        """Generate frames using file-based communication"""
        frame_counter = 0
        last_detection_time = 0
        
        while True:
            try:
                frame_counter += 1
                
                # Check if we have a current frame file
                if os.path.exists(self.temp_image_file):
                    try:
                        # Read current frame from file
                        current_frame = cv2.imread(self.temp_image_file)
                        
                        if current_frame is not None:
                            # Decide whether to run detection
                            current_time = time.time()
                            
                            if frame_counter < 30:
                                # First 30 frames: just display camera feed
                                display_frame = current_frame.copy()
                                cv2.putText(display_frame, f"Initializing... {frame_counter}/30", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                cv2.putText(display_frame, "File-based separation", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            
                            elif current_time - last_detection_time > 2.0:  # Run detection every 2 seconds
                                print(f"Running detection on frame {frame_counter}")
                                
                                # Process using file-based method
                                result_frame, success, message = self.detector.process_image_file(self.temp_image_file)
                                
                                if success and result_frame is not None:
                                    display_frame = result_frame
                                    cv2.putText(display_frame, "DETECTION SUCCESS", 
                                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    print("Detection successful!")
                                else:
                                    display_frame = current_frame.copy()
                                    cv2.putText(display_frame, "DETECTION FAILED", 
                                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.putText(display_frame, message[:30], 
                                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    print(f"Detection failed: {message}")
                                
                                last_detection_time = current_time
                            
                            else:
                                # Use previous frame or current frame
                                display_frame = current_frame.copy()
                                next_detection = 2.0 - (current_time - last_detection_time)
                                cv2.putText(display_frame, f"Next detection: {next_detection:.1f}s", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        else:
                            # Could not read frame file
                            display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(display_frame, "Cannot read frame file", 
                                       (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "FRAME ERROR", 
                                   (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(display_frame, str(e)[:30], 
                                   (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                else:
                    # No frame file available
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "Waiting for camera...", 
                               (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Encode and yield frame
                _, buffer = cv2.imencode('.jpg', display_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Generate frames error: {e}")
                # Emergency frame
                emergency_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(emergency_frame, "CRITICAL ERROR", 
                           (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', emergency_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        if not self.start_camera():
            return
        
        print(f"🔄 Separated web server: http://{host}:{port}")
        print("📁 Using file-based camera-DPU separation")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.is_running = False
            if self.cap:
                self.cap.release()
            
            # Cleanup temp files
            for temp_file in [self.temp_image_file, self.processed_image_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🔄 Initializing separated detector...")
    detector = SeparatedDetector(args.model)
    
    print("🌐 Starting separated web app...")
    app = SeparatedWebApp(detector, args.camera)
    
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
