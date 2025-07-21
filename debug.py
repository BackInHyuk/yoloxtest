#!/usr/bin/env python3
"""
Debug DPU Worker - 더미 데이터만 사용하여 segfault 원인 격리
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

class DebugDPUWorker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Communication files
        self.input_file = "/tmp/dpu_input.jpg"
        self.result_file = "/tmp/dpu_result.json"
        self.status_file = "/tmp/dpu_status.txt"
        
        # Test modes
        self.test_mode = "dummy"  # "dummy", "file", "real"
        
        # Load model
        self._load_model()
        
        # Write status
        with open(self.status_file, 'w') as f:
            f.write("DPU_READY")
        
        print("Debug DPU Worker ready")
    
    def _load_model(self):
        """Load DPU model - COMPLETE version from previous success"""
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs with FULL compatibility
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
            
            # Find DPU subgraph with all methods
            dpu_subgraph = None
            
            # Method 1: Device attribute
            for i, sg in enumerate(subgraphs):
                if sg is None:
                    continue
                try:
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = sg
                            print(f"Found DPU by device at index {i}")
                            break
                except:
                    continue
            
            # Method 2: Validation test
            if dpu_subgraph is None:
                for i, sg in enumerate(subgraphs):
                    if sg is None:
                        continue
                    try:
                        name = sg.get_name()
                        if name != "root":
                            test_runner = vart.Runner.create_runner(sg, "run")
                            if test_runner is not None:
                                dpu_subgraph = sg
                                print(f"Found DPU by validation at index {i}")
                                break
                    except:
                        continue
            
            if dpu_subgraph is None:
                raise ValueError("No DPU subgraph found")
            
            # Create runner
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            print("✅ Debug DPU loaded successfully")
            
        except Exception as e:
            print(f"Debug DPU model loading failed: {e}")
            sys.exit(1)
    
    def create_dummy_data(self):
        """Create dummy data like successful minimal_dpu_test.py"""
        try:
            input_shape = tuple(self.input_tensors[0].dims)  # (1, 416, 416, 3)
            
            # Method that worked in minimal_dpu_test.py
            dummy_data = np.random.randint(-128, 127, input_shape, dtype=np.int8)
            dummy_data = np.ascontiguousarray(dummy_data)
            
            print(f"Created dummy data: {dummy_data.shape}, {dummy_data.dtype}")
            return dummy_data
            
        except Exception as e:
            print(f"Error creating dummy data: {e}")
            return None
    
    def process_with_dummy_data(self):
        """Process using dummy data (known to work)"""
        try:
            print("🧪 Testing with DUMMY data (like minimal_dpu_test.py)")
            start_time = time.time()
            
            # Create dummy input (proven method)
            input_data = self.create_dummy_data()
            if input_data is None:
                return {"success": False, "error": "Failed to create dummy data"}
            
            # Prepare outputs (proven method)
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # Clear memory (proven method)
            gc.collect()
            
            print(">>> DPU inference with DUMMY data <<<")
            
            # DPU inference (exact proven method)
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            print(">>> DPU inference with DUMMY data completed <<<")
            
            # Simple count
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
                "method": "dummy_data",
                "timestamp": time.time()
            }
            
            print(f"✅ DUMMY data test SUCCESS: {result}")
            return result
            
        except Exception as e:
            print(f"❌ DUMMY data test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Dummy test failed: {str(e)}",
                "method": "dummy_data",
                "timestamp": time.time()
            }
    
    def process_real_image(self, image_path):
        """Process real image (this is where segfault likely occurs)"""
        try:
            print("🖼️ Testing with REAL image data")
            
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                return {"success": False, "error": "Could not read image"}
            
            print(f"Read real image: {frame.shape}, {frame.dtype}")
            start_time = time.time()
            
            # Preprocess (proven Method 2)
            img_h, img_w = frame.shape[:2]
            scale = min(416 / img_w, 416 / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            print(f"Resizing: {img_w}x{img_h} -> {new_w}x{new_h}")
            resized = cv2.resize(frame, (new_w, new_h))
            
            print("Creating padded array...")
            padded = np.full((416, 416, 3), 114, dtype=np.uint8)
            pad_x = (416 - new_w) // 2
            pad_y = (416 - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            print("Converting to int8...")
            # Convert to int8 (proven method)
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)
            
            print(f"Preprocessed real image: {input_data.shape}, {input_data.dtype}, range=[{input_data.min()}, {input_data.max()}]")
            
            # Prepare outputs
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # Clear memory
            gc.collect()
            
            print(">>> DPU inference with REAL image <<<")
            
            # DPU inference (proven method)
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            print(">>> DPU inference with REAL image completed <<<")
            
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
                "method": "real_image",
                "timestamp": time.time()
            }
            
            print(f"✅ REAL image test SUCCESS: {result}")
            return result
            
        except Exception as e:
            print(f"❌ REAL image test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Real image test failed: {str(e)}",
                "method": "real_image",
                "timestamp": time.time()
            }
    
    def run(self):
        """Main debug worker loop"""
        print("Debug DPU Worker started - testing different data types...")
        
        test_count = 0
        dummy_success = 0
        real_success = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Check for input every 2 seconds
                if os.path.exists(self.input_file):
                    test_count += 1
                    print(f"\n=== Test {test_count} ===")
                    
                    # Alternate between dummy and real data tests
                    if test_count % 2 == 1:
                        # Test with dummy data first
                        result = self.process_with_dummy_data()
                        if result["success"]:
                            dummy_success += 1
                    else:
                        # Test with real image
                        result = self.process_real_image(self.input_file)
                        if result["success"]:
                            real_success += 1
                    
                    # Write result
                    with open(self.result_file, 'w') as f:
                        json.dump(result, f)
                    
                    # Remove input file
                    try:
                        os.remove(self.input_file)
                    except:
                        pass
                    
                    print(f"Statistics: Dummy {dummy_success}/{(test_count+1)//2}, Real {real_success}/{test_count//2}")
                
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                print("Debug DPU Worker stopping...")
                break
            except Exception as e:
                print(f"Debug DPU Worker error: {e}")
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
    
    worker = DebugDPUWorker(args.model)
    worker.run()
