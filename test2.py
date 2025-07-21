#!/usr/bin/env python3
"""
Minimal detection using PIL instead of OpenCV
Testing if OpenCV is the cause of segfaults
"""

import numpy as np
import time
import gc
import argparse
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template_string
import io
import threading

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class MinimalPILDetector:
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
            
            # Get subgraphs with full compatibility
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
                else:
                    subgraphs = [root_subgraph]
            except:
                subgraphs = [root_subgraph]
            
            # Find DPU subgraph
            dpu_subgraph = None
            for sg in subgraphs:
                if sg is None:
                    continue
                try:
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = sg
                            break
                except:
                    continue
            
            if dpu_subgraph is None:
                for sg in subgraphs:
                    if sg is not None:
                        try:
                            if sg.get_name() != "root":
                                test_runner = vart.Runner.create_runner(sg, "run")
                                if test_runner is not None:
                                    dpu_subgraph = sg
                                    break
                        except:
                            continue
            
            if dpu_subgraph is None:
                raise ValueError("No DPU subgraph found")
            
            # Create runner
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            print("✅ DPU loaded (PIL mode)")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            exit(1)
    
    def preprocess_with_pil(self, pil_image):
        """Preprocess using PIL instead of OpenCV"""
        try:
            # Resize with PIL
            original_size = pil_image.size
            scale = min(416 / original_size[0], 416 / original_size[1])
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create padded image
            padded = Image.new('RGB', (416, 416), color=(114, 114, 114))
            paste_x = (416 - new_size[0]) // 2
            paste_y = (416 - new_size[1]) // 2
            padded.paste(resized, (paste_x, paste_y))
            
            # Convert to numpy
            np_image = np.array(padded)
            
            # Convert to int8 (proven method)
            input_data = (np_image.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)
            
            return input_data, scale, paste_x, paste_y
            
        except Exception as e:
            print(f"PIL preprocessing error: {e}")
            raise e
    
    def detect_with_pil(self, pil_image):
        """Detection using PIL preprocessing"""
        try:
            start_time = time.time()
            
            # Preprocess with PIL
            input_data, scale, pad_x, pad_y = self.preprocess_with_pil(pil_image)
            
            # Prepare outputs
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # Clear memory
            gc.collect()
            
            print("Starting DPU inference (PIL mode)...")
            
            # DPU inference
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            print("DPU inference completed (PIL mode)")
            
            # Count detections
            detection_count = 0
            for output in output_arrays:
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    objectness = output[0, :, :, 4]
                    detection_count += np.sum(objectness > 0.3)
            
            self.inference_time = time.time() - start_time
            self.detection_count = min(detection_count, 50)
            
            return True, f"PIL Success: {self.detection_count} detections"
            
        except Exception as e:
            error_msg = f"PIL detection error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg

class SyntheticImageApp:
    def __init__(self, detector):
        self.detector = detector
        self.app = Flask(__name__)
        self.app.config['THREADED'] = False
        
        # Synthetic image generation
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_result = "Waiting..."
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🧪 PIL-only Detection Test</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #0d1117;
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
            border: 2px solid #21262d;
            border-radius: 10px;
        }
        .status {
            margin: 20px 0;
            font-size: 18px;
            color: #58a6ff;
        }
        .test-mode {
            color: #f85149;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 PIL-only Detection Test</h1>
        <div class="status test-mode">
            NO OpenCV - Testing if OpenCV causes segfaults
        </div>
        <img src="/video_feed" alt="Synthetic Images" />
        <div class="status">
            Using PIL + Synthetic Images Only
        </div>
    </div>
</body>
</html>
            ''')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def create_synthetic_image(self):
        """Create synthetic image using PIL"""
        try:
            # Create colorful synthetic image
            img = Image.new('RGB', (640, 480), color=(50, 100, 150))
            draw = ImageDraw.Draw(img)
            
            # Add some shapes
            import random
            for i in range(5):
                x1 = random.randint(50, 500)
                y1 = random.randint(50, 400)
                x2 = x1 + random.randint(50, 100)
                y2 = y1 + random.randint(50, 100)
                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Add frame counter
            try:
                draw.text((10, 10), f"Frame: {self.frame_count}", fill=(255, 255, 255))
                draw.text((10, 30), f"PIL Mode", fill=(255, 255, 0))
                draw.text((10, 50), f"Result: {self.detection_result}", fill=(0, 255, 0))
            except:
                pass  # Skip if no font available
            
            return img
            
        except Exception as e:
            print(f"Synthetic image creation error: {e}")
            # Fallback: simple image
            img = Image.new('RGB', (640, 480), color=(100, 100, 100))
            return img
    
    def generate_frames(self):
        """Generate frames with synthetic images"""
        while True:
            try:
                self.frame_count += 1
                current_time = time.time()
                
                # Create synthetic image
                pil_image = self.create_synthetic_image()
                
                # Test detection every 5 seconds after frame 20
                if self.frame_count > 20 and current_time - self.last_detection_time > 5.0:
                    print(f"\n=== Testing PIL detection on frame {self.frame_count} ===")
                    
                    try:
                        success, message = self.detector.detect_with_pil(pil_image)
                        
                        if success:
                            self.detection_result = f"✅ {message}"
                            print(f"SUCCESS: {message}")
                        else:
                            self.detection_result = f"❌ {message[:30]}"
                            print(f"FAILED: {message}")
                            
                    except Exception as e:
                        self.detection_result = f"💥 Exception: {str(e)[:20]}"
                        print(f"EXCEPTION: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    self.last_detection_time = current_time
                
                # Add status overlay
                draw = ImageDraw.Draw(pil_image)
                try:
                    if self.frame_count <= 20:
                        draw.text((10, 70), f"Initializing: {self.frame_count}/20", fill=(255, 255, 0))
                    else:
                        next_test = 5.0 - (current_time - self.last_detection_time)
                        if next_test > 0:
                            draw.text((10, 70), f"Next test: {next_test:.1f}s", fill=(255, 255, 255))
                        else:
                            draw.text((10, 70), "Testing now...", fill=(255, 0, 0))
                except:
                    pass
                
                # Convert PIL to JPEG bytes
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format='JPEG', quality=80)
                img_bytes = img_buffer.getvalue()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
                
                time.sleep(0.5)  # 2 FPS
                
            except Exception as e:
                print(f"Frame generation error: {e}")
                # Emergency frame
                emergency_img = Image.new('RGB', (640, 480), color=(255, 0, 0))
                draw = ImageDraw.Draw(emergency_img)
                try:
                    draw.text((10, 10), "FRAME ERROR", fill=(255, 255, 255))
                    draw.text((10, 30), str(e)[:50], fill=(255, 255, 255))
                except:
                    pass
                
                img_buffer = io.BytesIO()
                emergency_img.save(img_buffer, format='JPEG')
                img_bytes = img_buffer.getvalue()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
                time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        print(f"🧪 PIL-only test server: http://{host}:{port}")
        print("🚫 NO OpenCV used - Testing if OpenCV causes segfaults")
        print("🎨 Using synthetic images only")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=False)
        except KeyboardInterrupt:
            print("\nShutting down...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🧪 Initializing PIL-only detector...")
    detector = MinimalPILDetector(args.model)
    
    print("🌐 Starting PIL test app...")
    app = SyntheticImageApp(detector)
    
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
