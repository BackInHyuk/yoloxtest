#!/usr/bin/env python3
"""
Simple single-threaded YOLOX detection test (no Flask, no threading)
"""

import cv2
import numpy as np
import time
import argparse

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class SimpleYOLOXDetector:
    def __init__(self, model_path: str, classes_file: str):
        self.model_path = model_path
        self.input_width = 416
        self.input_height = 416
        self.conf_threshold = 0.3
        self.nms_threshold = 0.45
        
        # Load class names
        self.class_names = self._load_classes(classes_file)
        
        # Load DPU model
        self._load_model()
        
        # Generate colors
        self.colors = self._generate_colors(len(self.class_names))
    
    def _load_classes(self, classes_file: str):
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            print(f"Loaded classes: {len(classes)}")
            return classes
        except FileNotFoundError:
            print(f"Class file not found: {classes_file}")
            return [f"class_{i}" for i in range(80)]
    
    def _load_model(self):
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            # Get all subgraphs - based on debug output, we need subgraph 1 (DPU)
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    subgraphs = root_subgraph.children_topological_sort()
                elif hasattr(root_subgraph, 'get_children'):
                    subgraphs = root_subgraph.get_children()
                else:
                    subgraphs = [root_subgraph]
            except:
                subgraphs = [root_subgraph]
            
            print(f"Found {len(subgraphs)} subgraphs:")
            for i, sg in enumerate(subgraphs):
                device = "Unknown"
                if sg.has_attr("device"):
                    device = sg.get_attr("device")
                print(f"  Subgraph {i}: {sg.get_name()} (device: {device})")
            
            # Based on debug output:
            # - Subgraph 0: YOLOX__YOLOX_QuantStub_quant_in__input_1_fix (device: CPU) 
            # - Subgraph 1: YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_0__inputs_3_fix (device: DPU)
            
            dpu_subgraph = None
            
            # First, try to find DPU subgraph by device attribute
            for i, subgraph in enumerate(subgraphs):
                try:
                    if subgraph.has_attr("device"):
                        device = subgraph.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = subgraph
                            print(f"Found DPU subgraph at index {i}")
                            break
                except:
                    continue
            
            # If not found by device attribute, use subgraph 1 based on debug output
            if dpu_subgraph is None:
                if len(subgraphs) > 1:
                    dpu_subgraph = subgraphs[1]  # Subgraph 1 is DPU based on debug
                    print("Using subgraph 1 as DPU (based on debug output)")
                else:
                    print("ERROR: Expected at least 2 subgraphs, but found only", len(subgraphs))
                    exit(1)
            
            # Verify this is not the root subgraph
            if dpu_subgraph.get_name() == "root":
                print("ERROR: Selected subgraph is 'root' which is not compiled!")
                print("Available subgraphs:")
                for i, sg in enumerate(subgraphs):
                    print(f"  {i}: {sg.get_name()}")
                exit(1)
            
            print(f"Using DPU subgraph: {dpu_subgraph.get_name()}")
            
            # Create DPU runner
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            print("DPU model loaded successfully")
            print(f"Input: {self.input_tensors[0].dims}")
            print(f"Outputs: {[t.dims for t in self.output_tensors]}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    def _generate_colors(self, num_classes: int):
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def preprocess(self, image: np.ndarray):
        img_h, img_w = image.shape[:2]
        scale = min(self.input_width / img_w, self.input_height / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - new_w) // 2
        pad_y = (self.input_height - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Convert to int8
        input_data = (padded.astype(np.int16) - 128).astype(np.int8)
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data, scale, pad_x, pad_y
    
    def detect(self, image: np.ndarray):
        """Simple detection without complex postprocessing"""
        try:
            print("Starting detection...")
            
            # Preprocess
            input_data, scale, pad_x, pad_y = self.preprocess(image)
            print(f"Preprocessed: {input_data.shape}, {input_data.dtype}")
            
            # Prepare outputs
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_arrays.append(output_array)
                print(f"Output array prepared: {shape}")
            
            print("Running DPU inference...")
            
            # DPU inference
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            print("DPU inference completed!")
            
            # Simple postprocessing - just check if we got outputs
            total_detections = 0
            for i, output in enumerate(output_arrays):
                print(f"Output {i}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
                # Count potential detections (very simple)
                if len(output.shape) == 4:
                    batch, h, w, features = output.shape
                    if features >= 85:
                        objectness = output[0, :, :, 4]  # objectness score
                        detections = np.sum(objectness > 0.1)
                        total_detections += detections
                        print(f"  Potential detections: {detections}")
            
            print(f"Total potential detections: {total_detections}")
            return True, total_detections
            
        except Exception as e:
            print(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False, 0
    
    def draw_simple_info(self, image, success, detection_count):
        """Draw simple info on image"""
        result = image.copy()
        
        if success:
            text = f"Detection OK - Found: {detection_count}"
            color = (0, 255, 0)
        else:
            text = "Detection FAILED"
            color = (0, 0, 255)
        
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    parser.add_argument('--classes', default='coco2017_classes.txt')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    print("=== Simple YOLOX Detection Test ===")
    print("This is a minimal test without Flask or threading")
    
    # Initialize detector
    detector = SimpleYOLOXDetector(args.model, args.classes)
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame")
                break
            
            frame_count += 1
            
            # Run detection every 10 frames to avoid overload
            if frame_count % 10 == 0:
                print(f"\n--- Frame {frame_count} ---")
                success, detection_count = detector.detect(frame)
                
                # Draw info
                result_frame = detector.draw_simple_info(frame, success, detection_count)
            else:
                result_frame = frame.copy()
                cv2.putText(result_frame, f"Frame {frame_count} (processing every 10th)", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Simple YOLOX Test', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'frame_{frame_count}.jpg', result_frame)
                print(f"Saved frame_{frame_count}.jpg")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    main()
