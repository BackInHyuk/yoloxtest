#!/usr/bin/env python3
"""
Minimal DPU test with dummy data - no camera involved
"""

import numpy as np
import time
import traceback

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

def test_dpu_minimal():
    """Test DPU with minimal dummy data"""
    
    print("=== Minimal DPU Test ===")
    print("Testing DPU inference with dummy data (no camera)")
    
    try:
        # Load model
        print("1. Loading model...")
        graph = xir.Graph.deserialize("yolox_nano_pt.xmodel")
        root_subgraph = graph.get_root_subgraph()
        
        # Get subgraphs
        subgraphs = []
        try:
            if hasattr(root_subgraph, 'children_topological_sort'):
                subgraphs = root_subgraph.children_topological_sort()
            else:
                subgraphs = [root_subgraph]
        except:
            subgraphs = [root_subgraph]
        
        print(f"Found {len(subgraphs)} subgraphs")
        
        # Find DPU subgraph
        dpu_subgraph = None
        for i, sg in enumerate(subgraphs):
            if sg.has_attr("device"):
                device = sg.get_attr("device")
                print(f"Subgraph {i}: {sg.get_name()} (device: {device})")
                if isinstance(device, str) and device.upper() == "DPU":
                    dpu_subgraph = sg
                    break
        
        if dpu_subgraph is None and len(subgraphs) > 1:
            dpu_subgraph = subgraphs[1]
            print("Using subgraph 1 as fallback")
        
        print(f"Selected DPU subgraph: {dpu_subgraph.get_name()}")
        
        # Create runner
        print("2. Creating DPU runner...")
        dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
        
        # Get tensors
        input_tensors = dpu_runner.get_input_tensors()
        output_tensors = dpu_runner.get_output_tensors()
        
        print(f"Input tensors: {len(input_tensors)}")
        print(f"Output tensors: {len(output_tensors)}")
        
        # Print tensor details
        for i, tensor in enumerate(input_tensors):
            print(f"Input {i}: shape={tensor.dims}, dtype={tensor.dtype}")
        
        for i, tensor in enumerate(output_tensors):
            print(f"Output {i}: shape={tensor.dims}, dtype={tensor.dtype}")
        
        # Test different input data types
        input_shape = tuple(input_tensors[0].dims)
        
        test_cases = [
            ("zeros_int8", lambda: np.zeros(input_shape, dtype=np.int8)),
            ("zeros_uint8", lambda: np.zeros(input_shape, dtype=np.uint8)),
            ("random_int8", lambda: np.random.randint(-128, 127, input_shape, dtype=np.int8)),
            ("converted_uint8_to_int8", lambda: (np.random.randint(0, 256, input_shape, dtype=np.uint8).astype(np.int16) - 128).astype(np.int8)),
        ]
        
        for test_name, data_gen in test_cases:
            print(f"\n3. Testing with {test_name}...")
            
            try:
                # Generate input data
                input_data = data_gen()
                print(f"   Input shape: {input_data.shape}")
                print(f"   Input dtype: {input_data.dtype}")
                print(f"   Input range: [{input_data.min()}, {input_data.max()}]")
                
                # Prepare outputs
                print("   Preparing output arrays...")
                output_arrays = []
                for j, tensor in enumerate(output_tensors):
                    shape = tuple(tensor.dims)
                    output_array = np.zeros(shape, dtype=np.float32)
                    output_arrays.append(output_array)
                    print(f"     Output {j}: shape={shape}, dtype={output_array.dtype}")
                
                # Critical point - DPU inference
                print("   >>> STARTING DPU INFERENCE <<<")
                start_time = time.time()
                
                job_id = dpu_runner.execute_async([input_data], output_arrays)
                dpu_runner.wait(job_id)
                
                inference_time = time.time() - start_time
                print(f"   >>> DPU INFERENCE COMPLETED in {inference_time:.3f}s <<<")
                
                # Check outputs
                for j, output in enumerate(output_arrays):
                    print(f"   Output {j}: range=[{output.min():.6f}, {output.max():.6f}], mean={output.mean():.6f}")
                
                print(f"   ✓ SUCCESS: {test_name}")
                
                # If we get here, this data type works!
                return True, test_name
                
            except Exception as e:
                print(f"   ✗ FAILED: {test_name}")
                print(f"   Error: {e}")
                print(f"   Traceback:")
                traceback.print_exc()
                print()
                continue
        
        return False, "all_failed"
        
    except Exception as e:
        print(f"Model loading/setup failed: {e}")
        traceback.print_exc()
        return False, "setup_failed"

def main():
    print("Starting minimal DPU test...")
    print("This test uses dummy data to isolate DPU inference issues\n")
    
    success, result = test_dpu_minimal()
    
    print("\n" + "="*50)
    if success:
        print(f"✓ DPU TEST PASSED with data type: {result}")
        print("The DPU hardware and driver are working correctly.")
        print("The issue might be in camera data preprocessing.")
    else:
        print(f"✗ DPU TEST FAILED: {result}")
        if result == "setup_failed":
            print("Problem with model loading or DPU setup.")
        else:
            print("Problem with DPU inference - possible driver/hardware issue.")
        print("Check DPU driver status and model compilation.")
    print("="*50)

if __name__ == "__main__":
    main()
