#!/usr/bin/env python3
"""
KV260 YOLOX Webcam Real-time Detection with Web Streaming
Access from host PC: http://board_ip:5000
"""

import cv2
import numpy as np
import time
import threading
import argparse
from flask import Flask, Response, render_template_string
from typing import List, Tuple, Optional
import json

# Vitis AI DPU runtime imports
try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class YOLOXDetector:
    def __init__(self, model_path: str, classes_file: str):
        """Initialize YOLOX detector"""
        self.model_path = model_path
        self.input_width = 416
        self.input_height = 416
        self.conf_threshold = 0.3
        self.nms_threshold = 0.45
        
        # Load class names
        self.class_names = self._load_classes(classes_file)
        
        # Load DPU model
        self._load_model()
        
        # Generate color palette
        self.colors = self._generate_colors(len(self.class_names))
        
        # Statistics
        self.fps = 0
        self.inference_time = 0
        self.detection_count = 0
    
    def _load_classes(self, classes_file: str) -> List[str]:
        """Load class names file"""
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"Loaded classes: {len(classes)}")
            return classes
        except FileNotFoundError:
            print(f"Class file not found: {classes_file}")
            return [f"class_{i}" for i in range(80)]  # COCO default 80 classes
    
    def _load_model(self):
        """Load DPU model with compatibility fix"""
        try:
            # Load xmodel
            graph = xir.Graph.deserialize(self.model_path)
            
            # Get root subgraph
            root_subgraph = graph.get_root_subgraph()
            
            # Try different methods to get subgraphs (compatibility fix)
            subgraphs = []
            try:
                # New API
                if hasattr(root_subgraph, 'children_topological_sort'):
                    subgraphs = root_subgraph.children_topological_sort()
                elif hasattr(root_subgraph, 'get_children'):
                    # Alternative API
                    subgraphs = root_subgraph.get_children()
                elif hasattr(root_subgraph, 'children'):
                    # Direct children access
                    subgraphs = root_subgraph.children
                else:
                    # Fallback: use root subgraph directly
                    subgraphs = [root_subgraph]
            except AttributeError as e:
                print(f"Subgraph API error: {e}")
                # Use root subgraph as fallback
                subgraphs = [root_subgraph]
            
            # Find DPU subgraph
            dpu_subgraph = None
            for subgraph in subgraphs:
                try:
                    if subgraph.has_attr("device"):
                        device_attr = subgraph.get_attr("device")
                        if isinstance(device_attr, str) and device_attr.upper() == "DPU":
                            dpu_subgraph = subgraph
                            break
                        elif hasattr(device_attr, 'upper') and device_attr.upper() == "DPU":
                            dpu_subgraph = subgraph
                            break
                except:
                    continue
            
            # If no DPU subgraph found, try using the first subgraph
            if dpu_subgraph is None:
                print("DPU subgraph not found, using first available subgraph")
                if subgraphs:
                    dpu_subgraph = subgraphs[0]
                else:
                    dpu_subgraph = root_subgraph
            
            # Create DPU runner
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            # Get input/output tensor info
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            print("DPU model loaded successfully")
            print(f"Input tensors: {len(self.input_tensors)}")
            print(f"Output tensors: {len(self.output_tensors)}")
            
            # Print tensor shapes for debugging
            for i, tensor in enumerate(self.input_tensors):
                print(f"Input {i}: {tensor.name}, shape: {tensor.dims}")
            for i, tensor in enumerate(self.output_tensors):
                print(f"Output {i}: {tensor.name}, shape: {tensor.dims}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Image preprocessing with proper tensor formatting for int8 quantized model"""
        img_h, img_w = image.shape[:2]
        
        # Get actual input tensor shape from DPU
        input_tensor = self.input_tensors[0]
        tensor_shape = tuple(input_tensor.dims)
        
        # Extract dimensions - NHWC format: [N, H, W, C]
        batch_size, tensor_h, tensor_w, channels = tensor_shape
        self.input_height, self.input_width = tensor_h, tensor_w
        
        print(f"Input tensor shape: {tensor_shape} (NHWC format)")
        print(f"Target size: {self.input_width}x{self.input_height}")
        
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Add padding
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # For int8 quantized models, keep as uint8 and add batch dimension
        # No normalization needed - DPU handles quantization internally
        input_data = padded.astype(np.uint8)  # Keep as uint8 for int8 model
        input_data = np.expand_dims(input_data, axis=0)   # Add batch dimension -> NHWC
        
        print(f"Preprocessed data shape: {input_data.shape}")
        print(f"Data type: {input_data.dtype}")
        print(f"Data range: [{input_data.min()}, {input_data.max()}]")
        
        return input_data, scale, pad_x, pad_y
    
    def postprocess(self, outputs: List[np.ndarray], orig_shape: Tuple[int, int], 
                   scale: float, pad_x: int, pad_y: int) -> List[dict]:
        """Post-processing for YOLOX multi-scale outputs"""
        try:
            all_boxes = []
            all_scores = []
            all_class_ids = []
            
            # YOLOX has 3 output scales: 52x52, 26x26, 13x13
            # Each output format: [batch, height, width, 85] where 85 = 4(bbox) + 1(obj) + 80(classes)
            
            for output_idx, output in enumerate(outputs):
                if len(output.shape) == 4:
                    batch_size, grid_h, grid_w, num_anchors = output.shape
                    output = output[0]  # Remove batch dimension
                else:
                    grid_h, grid_w, num_anchors = output.shape
                
                print(f"Processing output {output_idx}: shape {output.shape}")
                
                # Generate grid
                for i in range(grid_h):
                    for j in range(grid_w):
                        prediction = output[i, j, :]
                        
                        if len(prediction) < 85:
                            continue
                        
                        # Extract components
                        x_center = prediction[0]
                        y_center = prediction[1] 
                        width = prediction[2]
                        height = prediction[3]
                        objectness = prediction[4]
                        class_probs = prediction[5:85]  # 80 classes
                        
                        if objectness < self.conf_threshold:
                            continue
                        
                        # Get best class
                        class_id = np.argmax(class_probs)
                        class_prob = class_probs[class_id]
                        final_score = objectness * class_prob
                        
                        if final_score < self.conf_threshold:
                            continue
                        
                        # Convert to absolute coordinates
                        # Grid-based coordinate conversion
                        stride = 416 // grid_h  # 8, 16, or 32
                        
                        abs_x = (j + x_center) * stride
                        abs_y = (i + y_center) * stride
                        abs_w = np.exp(width) * stride
                        abs_h = np.exp(height) * stride
                        
                        # Convert to corner coordinates
                        x1 = (abs_x - abs_w/2 - pad_x) / scale
                        y1 = (abs_y - abs_h/2 - pad_y) / scale
                        x2 = (abs_x + abs_w/2 - pad_x) / scale
                        y2 = (abs_y + abs_h/2 - pad_y) / scale
                        
                        # Clip to image bounds
                        x1 = max(0, min(x1, orig_shape[1]))
                        y1 = max(0, min(y1, orig_shape[0]))
                        x2 = max(0, min(x2, orig_shape[1]))
                        y2 = max(0, min(y2, orig_shape[0]))
                        
                        if x2 > x1 and y2 > y1:  # Valid box
                            all_boxes.append([x1, y1, x2, y2])
                            all_scores.append(final_score)
                            all_class_ids.append(class_id)
            
            # Apply NMS
            if len(all_boxes) > 0:
                indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.conf_threshold, self.nms_threshold)
                if len(indices) > 0:
                    if isinstance(indices, np.ndarray):
                        indices = indices.flatten()
                    
                    detections = []
                    for i in indices:
                        detections.append({
                            'bbox': all_boxes[i],
                            'score': all_scores[i],
                            'class_id': all_class_ids[i],
                            'class_name': self.class_names[all_class_ids[i]] if all_class_ids[i] < len(self.class_names) else f"class_{all_class_ids[i]}"
                        })
                    
                    print(f"Final detections after NMS: {len(detections)}")
                    return detections
            
            return []
            
        except Exception as e:
            print(f"Postprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """Perform object detection with proper input handling for int8 model"""
        start_time = time.time()
        
        try:
            orig_shape = image.shape[:2]
            
            # Preprocessing
            input_data, scale, pad_x, pad_y = self.preprocess(image)
            
            # Create input arrays for DPU
            input_arrays = [input_data]  # Single input for YOLOX
            output_arrays = []
            
            # Prepare output arrays with correct shapes
            for tensor in self.output_tensors:
                tensor_shape = tuple(tensor.dims)
                output_array = np.zeros(tensor_shape, dtype=np.float32)
                output_arrays.append(output_array)
            
            # DPU inference
            job_id = self.dpu_runner.execute_async(input_arrays, output_arrays)
            self.dpu_runner.wait(job_id)
            
            # Post-processing
            detections = self.postprocess(output_arrays, orig_shape, scale, pad_x, pad_y)
            
            # Update statistics
            self.inference_time = time.time() - start_time
            self.detection_count = len(detections)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
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
            class_id = detection['class_id']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image

class WebStreamer:
    def __init__(self, detector: YOLOXDetector, camera_id: int = 0):
        self.detector = detector
        self.camera_id = camera_id
        self.cap = None
        self.frame = None
        self.result_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()
        
        # FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>KV260 YOLOX Real-time Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f0f0f0;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container { 
            text-align: center; 
            margin: 20px 0;
        }
        .stats { 
            display: flex; 
            justify-content: space-around; 
            margin: 20px 0;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }
        .stat-item { 
            text-align: center;
        }
        .stat-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: #007bff;
        }
        .stat-label { 
            font-size: 14px; 
            color: #666;
        }
        img { 
            max-width: 100%; 
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center; color: #333;">🚀 KV260 YOLOX Real-time Detection</h1>
        
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
        
        <div class="controls">
            <button onclick="toggleStream()">Toggle Stream</button>
            <button onclick="downloadSnapshot()">📸 Snapshot</button>
            <button onclick="refreshStats()">🔄 Refresh</button>
        </div>
    </div>

    <script>
        // Update statistics
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
        
        function toggleStream() {
            fetch('/toggle', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log('Stream toggled:', data.running);
                    updateStats();
                });
        }
        
        function downloadSnapshot() {
            window.open('/snapshot', '_blank');
        }
        
        function refreshStats() {
            updateStats();
        }
        
        // Update statistics periodically
        setInterval(updateStats, 1000);
        updateStats(); // Initial load
    </script>
</body>
</html>
            ''')
        
        @self.app.route('/video_feed')
        def video_feed():
            """Provide video stream"""
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def stats():
            """Statistics API"""
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            return {
                'fps': fps,
                'inference_time': self.detector.inference_time,
                'detections': self.detector.detection_count,
                'running': self.is_running,
                'frame_count': self.frame_count
            }
        
        @self.app.route('/toggle', methods=['POST'])
        def toggle():
            """Toggle stream"""
            if self.is_running:
                self.stop_camera()
            else:
                self.start_camera()
            return {'running': self.is_running}
        
        @self.app.route('/snapshot')
        def snapshot():
            """Current frame snapshot"""
            with self.lock:
                if self.result_frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.result_frame)
                    return Response(buffer.tobytes(), mimetype='image/jpeg')
            return "No frame available", 404
    
    def start_camera(self):
        """Start camera"""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_id}")
            return
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print("Camera started")
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Camera stopped")
    
    def generate_frames(self):
        """Frame generator"""
        while True:
            with self.lock:
                if self.result_frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.result_frame, 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Generate default image
                    default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(default_frame, "Camera Not Started", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', default_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    def process_frames(self):
        """Frame processing thread"""
        while True:
            if self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Perform detection
                    detections = self.detector.detect(frame)
                    
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
                    
                    with self.lock:
                        self.frame = frame
                        self.result_frame = result_frame
                    
                    self.frame_count += 1
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def run(self, host='0.0.0.0', port=5000):
        """Run web server"""
        # Start frame processing thread
        processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        processing_thread.start()
        
        # Auto-start camera
        self.start_camera()
        
        print(f"Web server starting: http://{host}:{port}")
        print("Press Ctrl+C to exit")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.stop_camera()

def main():
    parser = argparse.ArgumentParser(description='KV260 YOLOX Web Streaming Detector')
    parser.add_argument('--model', default='yolox_nano_pt.xmodel', help='YOLOX model file')
    parser.add_argument('--classes', default='coco2017_classes.txt', help='Class names file')
    parser.add_argument('--camera', type=int, default=0, help='Camera device number')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    
    args = parser.parse_args()
    
    # Initialize YOLOX detector
    print("Initializing YOLOX detector...")
    detector = YOLOXDetector(args.model, args.classes)
    
    # Initialize web streamer
    streamer = WebStreamer(detector, args.camera)
    
    # Run web server
    streamer.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
