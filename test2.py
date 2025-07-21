#!/usr/bin/env python3
"""
File-only YOLOX test - no camera involved at all
Creates synthetic images and tests DPU inference
"""

import numpy as np
import time
import cv2
import argparse

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class FileOnlyYOLOXDetector:
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
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    subgraphs = root_subgraph.children_topological_sort()
                    if subgraphs is None:
                        subgraphs = []
                elif hasattr(root_subgraph, 'get_children'):
                    subgraphs = root_subgraph.get_children()
                    if subgraphs is None:
                        subgraphs = []
                else:
                    subgraphs = [root_subgraph]
            except:
                subgraphs = [root_subgraph]
            
            if not subgraphs:
                raise ValueError("No subgraphs found")
            
            print(f"Found {len(subgraphs)} subgraphs:")
            
            # List all subgraphs for debugging
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
            
            # Find DPU subgraph - based on debug output, should be subgraph 2
            dpu_subgraph = None
            
            # Method 1: Find by device attribute
            for i, sg in enumerate(subgraphs):
                if sg is None:
                    continue
                try:
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = sg
                            print(f"Found DPU subgraph at index {i} by device attribute")
                            break
                except Exception as e:
                    print(f"Error checking subgraph {i}: {e}")
                    continue
            
            # Method 2: Based on debug output, try specific indices
            if dpu_subgraph is None:
                print("DPU subgraph not found by device attribute, trying known indices...")
                
                # From debug output, subgraph 2 has device DPU
                for idx in [2, 1]:  # Try subgraph 2 first, then 1
                    if len(subgraphs) > idx and subgraphs[idx] is not None:
                        try:
                            sg = subgraphs[idx]
                            name = sg.get_name()
                            print(f"Trying subgraph {idx}: {name}")
                            
                            # Avoid root subgraph
                            if name != "root":
                                # Try to create a runner to test if it's valid
                                try:
                                    test_runner = vart.Runner.create_runner(sg, "run")
                                    if test_runner is not None:
                                        dpu_subgraph = sg
                                        print(f"Successfully validated subgraph {idx} as DPU")
                                        break
                                except Exception as test_e:
                                    print(f"Subgraph {idx} failed runner test: {test_e}")
                                    continue
                            else:
                                print(f"Skipping subgraph {idx} (root)")
                        except Exception as e:
                            print(f"Error testing subgraph {idx}: {e}")
                            continue
            
            # Method 3: Try any non-root subgraph
            if dpu_subgraph is None:
                print("Trying any non-root subgraph...")
                for i, sg in enumerate(subgraphs):
                    if sg is None:
                        continue
                    try:
                        name = sg.get_name()
                        if name != "root":
                            try:
                                test_runner = vart.Runner.create_runner(sg, "run")
                                if test_runner is not None:
                                    dpu_subgraph = sg
                                    print(f"Found working subgraph at index {i}: {name}")
                                    break
                            except:
                                continue
                    except:
                        continue
            
            if dpu_subgraph is None:
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
                raise ValueError("No valid DPU subgraph found")
            
            # Create DPU runner
            print(f"Creating DPU runner with selected subgraph...")
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            
            if self.dpu_runner is None:
                raise ValueError("Failed to create DPU runner")
            
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            if not self.input_tensors or not self.output_tensors:
                raise ValueError("Failed to get tensor information")
            
            print("DPU model loaded successfully")
            print(f"Selected subgraph: {dpu_subgraph.get_name()}")
            print(f"Input: {self.input_tensors[0].dims}")
            print(f"Outputs: {[t.dims for t in self.output_tensors]}")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    def create_synthetic_image(self, width=640, height=480):
        """Create a synthetic image without using camera"""
        try:
            print(f"Creating synthetic image: {width}x{height}")
            
            # Create a realistic looking synthetic image
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some patterns to make it look realistic
            # Background gradient
            for y in range(height):
                for x in range(width):
                    image[y, x, 0] = min(255, (x * 255) // width)  # Red gradient
                    image[y, x, 1] = min(255, (y * 255) // height)  # Green gradient
                    image[y, x, 2] = 128  # Blue constant
            
            # Add some rectangles (like objects)
            cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
            cv2.rectangle(image, (300, 200), (400, 350), (0, 255, 0), -1)  # Green rectangle
            cv2.rectangle(image, (450, 50), (550, 150), (0, 0, 255), -1)   # Red rectangle
            
            # Add some noise
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            image = np.clip(image.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
            
            print(f"Synthetic image created: shape={image.shape}, dtype={image.dtype}")
            return image
            
        except Exception as e:
            print(f"Failed to create synthetic image: {e}")
            raise e
    
    def preprocess_safe(self, image: np.ndarray):
        """Ultra-safe preprocessing"""
        try:
            print(f"Starting preprocessing: {image.shape}, {image.dtype}")
            
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            img_h, img_w = image.shape[:2]
            
            # Calculate scale
            scale = min(self.input_width / img_w, self.input_height / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            print(f"Scaling: {img_w}x{img_h} -> {new_w}x{new_h} (scale: {scale:.3f})")
            
            # Resize using numpy instead of cv2 to avoid potential opencv issues
            try:
                # Manual resize using numpy to avoid OpenCV
                resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                
                for c in range(3):
                    for y in range(new_h):
                        for x in range(new_w):
                            orig_y = int(y / scale)
                            orig_x = int(x / scale)
                            orig_y = min(orig_y, img_h - 1)
                            orig_x = min(orig_x, img_w - 1)
                            resized[y, x, c] = image[orig_y, orig_x, c]
                
                print(f"Manual resize completed: {resized.shape}")
                
            except Exception as e:
                print(f"Manual resize failed, falling back to cv2: {e}")
                resized = cv2.resize(image, (new_w, new_h))
            
            # Create padded array
            padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
            pad_x = (self.input_width - new_w) // 2
            pad_y = (self.input_height - new_h) // 2
            
            # Place image
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # Convert to int8 very carefully
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            
            print(f"Preprocessing completed: {input_data.shape}, range=[{input_data.min()}, {input_data.max()}]")
            
            return input_data, scale, pad_x, pad_y
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise e
    
    def detect_safe(self, image: np.ndarray):
        """Ultra-safe detection"""
        try:
            print("=== Starting safe detection ===")
            
            # Preprocess
            input_data, scale, pad_x, pad_y = self.preprocess_safe(image)
            
            # Prepare outputs
            output_arrays = []
            for i, tensor in enumerate(self.output_tensors):
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_arrays.append(output_array)
                print(f"Output {i} prepared: {shape}")
            
            print(">>> Starting DPU inference <<<")
            
            # DPU inference
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            print(">>> DPU inference completed <<<")
            
            # Simple output check
            total_detections = 0
            for i, output in enumerate(output_arrays):
                print(f"Output {i}: range=[{output.min():.3f}, {output.max():.3f}]")
                # Very basic detection count
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    objectness = output[0, :, :, 4]
                    detections = np.sum(objectness > 0.1)
                    total_detections += detections
            
            print(f"Detection completed - found {total_detections} potential objects")
            return True, total_detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return False, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    parser.add_argument('--classes', default='coco2017_classes.txt')
    parser.add_argument('--tests', type=int, default=5, help='Number of test images')
    parser.add_argument('--save-images', action='store_true', help='Save test images')
    args = parser.parse_args()
    
    print("=== File-Only YOLOX Test ===")
    print("Testing with synthetic images (NO CAMERA)")
    
    detector = None
    
    try:
        # Initialize detector
        print("Initializing detector...")
        detector = FileOnlyYOLOXDetector(args.model, args.classes)
        
        print(f"Running {args.tests} tests with synthetic images...")
        
        success_count = 0
        
        for i in range(args.tests):
            print(f"\n--- Test {i+1}/{args.tests} ---")
            
            try:
                # Create synthetic image
                image = detector.create_synthetic_image()
                
                # Save image if requested
                if args.save_images:
                    filename = f'synthetic_test_{i+1}.jpg'
                    cv2.imwrite(filename, image)
                    print(f"Saved {filename}")
                
                # Run detection
                success, detection_count = detector.detect_safe(image)
                
                if success:
                    success_count += 1
                    print(f"✓ Test {i+1} SUCCESS - Detections: {detection_count}")
                else:
                    print(f"✗ Test {i+1} FAILED")
                
                # Small delay between tests
                time.sleep(0.5)
                
            except Exception as e:
                print(f"✗ Test {i+1} ERROR: {e}")
        
        print(f"\n=== Results ===")
        print(f"Total tests: {args.tests}")
        print(f"Successful: {success_count}")
        print(f"Success rate: {success_count/args.tests*100:.1f}%")
        
        if success_count == args.tests:
            print("🎉 ALL TESTS PASSED!")
            print("DPU inference works correctly with image data.")
            print("The segfault issue is likely related to camera/OpenCV interaction.")
        else:
            print("⚠️  Some tests failed - may indicate DPU or preprocessing issues.")
    
    except Exception as e:
        print(f"Test setup failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if detector is not None:
            detector.dpu_runner = None
            detector = None
        print("Cleanup completed")

if __name__ == "__main__":
    main()
