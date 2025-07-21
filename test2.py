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
        
        # Initialize as None for safe cleanup
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Load class names
        self.class_names = self._load_classes(classes_file)
        
        # Load DPU model
        self._load_model()
        
        # Generate colors
        self.colors = self._generate_colors(len(self.class_names))
    
    def __del__(self):
        """Safe cleanup when object is destroyed"""
        try:
            if hasattr(self, 'dpu_runner') and self.dpu_runner is not None:
                print("Cleaning up DPU runner...")
                # Don't explicitly delete - let Python handle it
                self.dpu_runner = None
        except:
            pass  # Ignore cleanup errors
    
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
        """Safe image preprocessing with extensive error checking"""
        try:
            if image is None:
                raise ValueError("Input image is None")
            
            if len(image.shape) != 3:
                raise ValueError(f"Expected 3D image, got shape: {image.shape}")
            
            img_h, img_w, img_c = image.shape
            
            if img_c != 3:
                raise ValueError(f"Expected 3 channels, got {img_c}")
            
            print(f"Input image: {img_w}x{img_h}x{img_c}, dtype={image.dtype}")
            
            # Calculate scale
            scale = min(self.input_width / img_w, self.input_height / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            print(f"Resizing to: {new_w}x{new_h} (scale: {scale:.3f})")
            
            # Resize safely
            try:
                resized = cv2.resize(image, (new_w, new_h))
                print(f"Resized shape: {resized.shape}")
            except Exception as e:
                raise RuntimeError(f"Resize failed: {e}")
            
            # Create padding array safely
            try:
                padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
                print(f"Padded array created: {padded.shape}")
            except Exception as e:
                raise RuntimeError(f"Padding array creation failed: {e}")
            
            # Calculate padding offsets
            pad_x = (self.input_width - new_w) // 2
            pad_y = (self.input_height - new_h) // 2
            
            print(f"Padding offsets: x={pad_x}, y={pad_y}")
            
            # Place resized image in padded array safely
            try:
                if pad_y + new_h > self.input_height or pad_x + new_w > self.input_width:
                    raise ValueError(f"Padding bounds exceeded: y={pad_y}+{new_h}>{self.input_height}, x={pad_x}+{new_w}>{self.input_width}")
                
                padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
                print(f"Image placed in padded array successfully")
            except Exception as e:
                raise RuntimeError(f"Image placement failed: {e}")
            
            # Convert to int8 safely (this is critical!)
            try:
                # Method 1: Safe conversion with bounds checking
                temp_data = padded.astype(np.int16)  # Use int16 to prevent overflow
                temp_data = temp_data - 128  # Shift to int8 range
                
                # Clamp to int8 range
                temp_data = np.clip(temp_data, -128, 127)
                
                # Convert to int8
                input_data = temp_data.astype(np.int8)
                
                print(f"Converted to int8: shape={input_data.shape}, range=[{input_data.min()}, {input_data.max()}]")
                
            except Exception as e:
                raise RuntimeError(f"Int8 conversion failed: {e}")
            
            # Add batch dimension safely
            try:
                input_data = np.expand_dims(input_data, axis=0)
                print(f"Added batch dimension: {input_data.shape}")
                
                # Verify final shape matches expected input
                expected_shape = (1, 416, 416, 3)
                if input_data.shape != expected_shape:
                    raise ValueError(f"Final shape {input_data.shape} != expected {expected_shape}")
                
            except Exception as e:
                raise RuntimeError(f"Batch dimension addition failed: {e}")
            
            # Verify data integrity
            if not np.isfinite(input_data).all():
                raise ValueError("Input data contains non-finite values")
            
            print(f"Preprocessing completed successfully")
            return input_data, scale, pad_x, pad_y
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise e
    
    def detect(self, image: np.ndarray):
        """Safe detection with extensive error checking"""
        try:
            print("Starting detection...")
            
            # Validate input image
            if image is None:
                raise ValueError("Input image is None")
            
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(image)}")
            
            print(f"Input image validation passed: {image.shape}, {image.dtype}")
            
            # Preprocess with error handling
            try:
                input_data, scale, pad_x, pad_y = self.preprocess(image)
                print(f"Preprocessing completed successfully")
            except Exception as e:
                print(f"Preprocessing failed: {e}")
                return False, 0
            
            # Prepare outputs safely
            try:
                output_arrays = []
                for i, tensor in enumerate(self.output_tensors):
                    shape = tuple(tensor.dims)
                    print(f"Preparing output {i}: shape={shape}")
                    
                    # Create output array with proper initialization
                    output_array = np.zeros(shape, dtype=np.float32)
                    
                    # Verify array was created properly
                    if output_array is None or output_array.size == 0:
                        raise RuntimeError(f"Failed to create output array {i}")
                    
                    output_arrays.append(output_array)
                    print(f"Output array {i} prepared successfully")
                    
            except Exception as e:
                print(f"Output preparation failed: {e}")
                return False, 0
            
            print("All output arrays prepared. Starting DPU inference...")
            
            # DPU inference with enhanced error handling
            try:
                # Validate input one more time before DPU call
                if input_data is None or input_data.size == 0:
                    raise ValueError("Input data is None or empty")
                
                if not np.isfinite(input_data).all():
                    raise ValueError("Input data contains non-finite values")
                
                print(f"Final input validation: shape={input_data.shape}, dtype={input_data.dtype}, range=[{input_data.min()}, {input_data.max()}]")
                
                # The critical DPU call
                print(">>> Calling execute_async <<<")
                job_id = self.dpu_runner.execute_async([input_data], output_arrays)
                
                print(">>> Calling wait <<<")
                self.dpu_runner.wait(job_id)
                
                print(">>> DPU inference completed successfully <<<")
                
            except Exception as e:
                print(f"DPU inference failed: {e}")
                import traceback
                traceback.print_exc()
                return False, 0
            
            # Simple postprocessing - just count potential detections
            try:
                total_detections = 0
                for i, output in enumerate(output_arrays):
                    if output is None:
                        continue
                        
                    print(f"Output {i}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
                    
                    # Very simple detection counting
                    if len(output.shape) == 4 and output.shape[-1] >= 85:
                        batch, h, w, features = output.shape
                        if features >= 85:
                            objectness = output[0, :, :, 4]  # objectness score
                            detections = np.sum(objectness > 0.1)
                            total_detections += detections
                            print(f"  Potential detections in output {i}: {detections}")
                
                print(f"Total potential detections: {total_detections}")
                return True, total_detections
                
            except Exception as e:
                print(f"Postprocessing error: {e}")
                # Even if postprocessing fails, DPU inference succeeded
                return True, 0
            
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
    parser.add_argument('--save-images', action='store_true', help='Save test images')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to test')
    args = parser.parse_args()
    
    print("=== Simple YOLOX Detection Test ===")
    print("This is a minimal test without Flask or threading")
    print("Running in headless mode (no GUI)")
    
    detector = None
    cap = None
    
    try:
        # Initialize detector
        print("Initializing detector...")
        detector = SimpleYOLOXDetector(args.model, args.classes)
        
        # Initialize camera
        print("Initializing camera...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Running detection test for {args.frames} frames...")
        print("Press Ctrl+C to quit early")
        
        frame_count = 0
        success_count = 0
        
        for test_frame in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame")
                break
            
            frame_count += 1
            
            # Run detection every frame for this test
            print(f"\n--- Frame {frame_count}/{args.frames} ---")
            success, detection_count = detector.detect(frame)
            
            if success:
                success_count += 1
                print(f"✓ Detection successful - Found: {detection_count}")
            else:
                print("✗ Detection failed")
            
            # Save image if requested
            if args.save_images and success:
                result_frame = detector.draw_simple_info(frame, success, detection_count)
                filename = f'test_frame_{frame_count}.jpg'
                cv2.imwrite(filename, result_frame)
                print(f"Saved {filename}")
            
            # Small delay to prevent overload
            time.sleep(0.5)
        
        print(f"\n=== Test Results ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Successful detections: {success_count}")
        if frame_count > 0:
            print(f"Success rate: {success_count/frame_count*100:.1f}%")
        print("Test completed successfully!")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Safe cleanup
        print("Cleaning up resources...")
        
        if cap is not None:
            try:
                cap.release()
                print("Camera released")
            except:
                pass
        
        if detector is not None:
            try:
                # Clear references to help with cleanup
                detector.dpu_runner = None
                detector = None
                print("Detector cleaned up")
            except:
                pass
        
        print("Cleanup completed - exiting safely")

if __name__ == "__main__":
    main()
