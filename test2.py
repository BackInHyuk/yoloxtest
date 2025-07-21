#!/usr/bin/env python3
"""
Stable YOLOX Web Detection with Fixed Memory Management
Uses the proven method from final_dpu_test.py
"""

import cv2
import numpy as np
import time
import threading
import argparse
import gc
from flask import Flask, Response, render_template_string
from typing import List, Tuple, Optional

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class StableYOLOXDetector:
    def __init__(self, model_path: str, classes_file: str):
        self.model_path = model_path
        self.input_width = 416
        self.input_height = 416
        self.conf_threshold = 0.3
        self.nms_threshold = 0.45
        
        # Initialize safely
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Load class names
        self.class_names = self._load_classes(classes_file)
        
        # Load DPU model
        self._load_model()
        
        # Generate colors
        self.colors = self._generate_colors(len(self.class_names))
        
        # Statistics
        self.fps = 0
        self.inference_time = 0
        self.detection_count = 0
    
    def _load_classes(self, classes_file: str) -> List[str]:
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"Loaded classes: {len(classes)}")
            return classes
        except FileNotFoundError:
            print(f"Class file not found: {classes_file}")
            return [f"class_{i}" for i in range(80)]
    
    def _load_model(self):
        """Load DPU model using proven stable method"""
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs with XIR API compatibility
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    children = root_subgraph.children_topological_sort()
                    # Handle both list and set return types
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
                    subgraphs = [root_subgraph]
            except Exception as e:
                print(f"Error getting subgraphs: {e}")
                subgraphs = [root_subgraph]
            
            if not subgraphs:
                raise ValueError("No subgraphs found")
            
            print(f"Found {len(subgraphs)} subgraphs")
            
            # Debug: print all subgraphs safely
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
                    print(f"  Subgraph {i}: Error getting info - {e}")
            
            # Find DPU subgraph using proven method (device attribute first)
            dpu_subgraph = None
            
            # Method 1: Find by device attribute (most reliable)
            for i, sg in enumerate(subgraphs):
                if sg is None:
                    continue
                try:
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = sg
                            print(f"✅ Found DPU subgraph at index {i} by device attribute")
                            break
                except Exception as e:
                    print(f"Error checking subgraph {i}: {e}")
                    continue
            
            # Method 2: Try to create runners for non-root subgraphs (validation test)
            if dpu_subgraph is None:
                print("DPU subgraph not found by device attribute, testing subgraphs...")
                for i, sg in enumerate(subgraphs):
                    if sg is None:
                        continue
                    try:
                        name = sg.get_name()
                        if name == "root":
                            print(f"  Skipping subgraph {i}: root")
                            continue
                        
                        # Test if this subgraph can create a runner
                        print(f"  Testing subgraph {i}: {name}")
                        test_runner = vart.Runner.create_runner(sg, "run")
                        if test_runner is not None:
                            dpu_subgraph = sg
                            print(f"✅ Validated subgraph {i} as working DPU: {name}")
                            break
                        
                    except Exception as e:
                        print(f"  Subgraph {i} failed validation: {e}")
                        continue
            
            # Method 3: Fallback to any non-root subgraph
            if dpu_subgraph is None:
                print("No validated subgraph found, using first non-root subgraph...")
                for i, sg in enumerate(subgraphs):
                    if sg is not None:
                        try:
                            name = sg.get_name()
                            if name != "root":
                                dpu_subgraph = sg
                                print(f"Using fallback subgraph {i}: {name}")
                                break
                        except:
                            continue
            
            if dpu_subgraph is None:
                raise ValueError("No valid DPU subgraph found")
            
            # Create DPU runner
            print("Creating DPU runner...")
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            if self.dpu_runner is None:
                raise ValueError("Failed to create DPU runner")
            
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            if not self.input_tensors or not self.output_tensors:
                raise ValueError("Failed to get tensor information")
            
            print("✅ DPU model loaded successfully")
            print(f"Selected subgraph: {dpu_subgraph.get_name()}")
            print(f"Input: {self.input_tensors[0].dims}")
            print(f"Outputs: {[t.dims for t in self.output_tensors]}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def preprocess_stable(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Stable preprocessing using proven memory management"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            img_h, img_w = image.shape[:2]
            
            # Calculate scale
            scale = min(self.input_width / img_w, self.input_height / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create padded array
            padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
            pad_x = (self.input_width - new_w) // 2
            pad_y = (self.input_height - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # CRITICAL: Use the proven stable conversion method
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            
            # CRITICAL: Ensure contiguous memory layout (this was the key!)
            input_data = np.ascontiguousarray(input_data)
            
            return input_data, scale, pad_x, pad_y
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise e
    
    def detect_stable(self, image: np.ndarray) -> List[dict]:
        """Stable detection using proven Method 2 from final test"""
        start_time = time.time()
        
        try:
            orig_shape = image.shape[:2]
            
            # Preprocessing with stable method
            input_data, scale, pad_x, pad_y = self.preprocess_stable(image)
            
            # CRITICAL: Manual output allocation with contiguous memory
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                # CRITICAL: Ensure contiguous memory (this prevents segfault!)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # CRITICAL: Clear memory before DPU call
            gc.collect()
            
            # PROVEN METHOD: execute_async + wait (Method 2 from test)
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            # Simple postprocessing - count detections
            detections = []
            total_detections = 0
            
            for i, output in enumerate(output_arrays):
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    batch, h, w, features = output.shape
                    if features >= 85:
                        objectness = output[0, :, :, 4]  # objectness score
                        high_conf = objectness > self.conf_threshold
                        detection_indices = np.where(high_conf)
                        
                        for y, x in zip(detection_indices[0], detection_indices[1]):
                            if objectness[y, x] > self.conf_threshold:
                                # Simple detection entry
                                detections.append({
                                    'bbox': [x*10, y*10, (x+1)*10, (y+1)*10],  # Dummy bbox
                                    'score': float(objectness[y, x]),
                                    'class_id': 0,
                                    'class_name': 'object'
                                })
                                total_detections += 1
                                
                                if total_detections >= 20:  # Limit detections
                                    break
                        if total_detections >= 20:
                            break
            
            # Update statistics
            self.inference_time = time.time() - start_time
            self.detection_count = len(detections)
            
            return detections[:10]  # Return max 10 detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            self.inference_time = time.time() - start_time
            self.detection_count = 0
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            score = detection['score']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)  # Green for all detections
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image

class StableWebStreamer:
    def __init__(self, detector: StableYOLOXDetector, camera_id: int = 0):
        self.detector = detector
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.result_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
        
        # FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Stable YOLOX Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .stats { 
            display: flex; 
            justify-content: space-around; 
            margin: 20px 0;
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
        }
        .stat-item { 
            text-align: center;
        }
        .stat-value { 
            font-size: 28px; 
            font-weight: bold; 
            color: #00ff88;
            text-shadow: 0 0 10px rgba(0,255,136,0.5);
        }
        .stat-label { 
            font-size: 14px; 
            color: #ccc;
        }
        img { 
            max-width: 100%; 
            border: 3px solid #00ff88;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,255,136,0.3);
        }
        .video-container { 
            text-align: center; 
            margin: 20px 0;
        }
        .status { 
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
        }
        .success { color: #00ff88; }
        .error { color: #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center;">🚀 Stable YOLOX Real-time Detection</h1>
        <p style="text-align: center;">✅ Memory-safe DPU inference | 🔧 Proven stable method</p>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="fps">--</div>
                <div class="stat-label">FPS</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="inference-time">--</div>
                <div class="stat-label">Inference (ms)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="detections">--</div>
                <div class="stat-label">Objects</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="status">🟢</div>
                <div class="stat-label">Status</div>
            </div>
        </div>
        
        <div class="video-container">
            <img src="/video_feed" alt="YOLOX Detection Stream" id="video-stream">
        </div>
        
        <div class="status success">
            ✅ Using proven stable memory management method
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('inference-time').textContent = (data.inference_time * 1000).toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('status').textContent = data.running ? '🟢' : '🔴';
                })
                .catch(error => console.error('Error:', error));
        }
        
        setInterval(updateStats, 1000);
        updateStats();
    </script>
</body>
</html>
            ''')
        
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
                'inference_time': self.detector.inference_time,
                'detections': self.detector.detection_count,
                'running': self.is_running
            }
    
    def start_camera(self):
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print("Camera started with stable detection")
    
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Camera stopped")
    
    def generate_frames(self):
        while True:
            with self.lock:
                if self.result_frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.result_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Default frame
                    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(default_frame, "Stable Detection Ready", (180, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    _, buffer = cv2.imencode('.jpg', default_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def process_frames(self):
        """Frame processing with stable detection"""
        while True:
            if self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    try:
                        # Stable detection
                        detections = self.detector.detect_stable(frame)
                        
                        # Draw results
                        result_frame = self.detector.draw_detections(frame, detections)
                        
                        # Add info overlay
                        current_time = time.time()
                        elapsed_time = current_time - self.start_time
                        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                        
                        cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Inference: {self.detector.inference_time*1000:.1f}ms", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Objects: {len(detections)}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(result_frame, "STABLE MODE", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        with self.lock:
                            self.frame = frame
                            self.result_frame = result_frame
                        
                        self.frame_count += 1
                        
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        # Continue running even if one frame fails
                        with self.lock:
                            self.result_frame = frame
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def run(self, host='0.0.0.0', port=5000):
        # Start frame processing thread
        processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        processing_thread.start()
        
        # Auto-start camera
        self.start_camera()
        
        print(f"🚀 Stable web server starting: http://{host}:{port}")
        print("✅ Using memory-safe DPU inference method")
        print("Press Ctrl+C to exit")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_camera()

def main():
    parser = argparse.ArgumentParser(description='Stable YOLOX Web Detection')
    parser.add_argument('--model', default='yolox_nano_pt.xmodel', help='YOLOX model file')
    parser.add_argument('--classes', default='coco2017_classes.txt', help='Class names file')
    parser.add_argument('--camera', type=int, default=0, help='Camera device number')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    
    args = parser.parse_args()
    
    print("🚀 Initializing Stable YOLOX Detector...")
    detector = StableYOLOXDetector(args.model, args.classes)
    
    print("🌐 Starting web streamer...")
    streamer = StableWebStreamer(detector, args.camera)
    
    streamer.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
