#!/usr/bin/env python3
"""
Simple YOLOX test script to isolate the data type issue
"""

import cv2
import numpy as np
import time

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

def test_yolox_simple():
    model_path = "yolox_nano_pt.xmodel"
    
    # Load model
    graph = xir.Graph.deserialize(model_path)
    root_subgraph = graph.get_root_subgraph()
    
    # Get subgraphs
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
    
    print(f"Found {len(subgraphs)} subgraphs")
    
    # Find DPU subgraph (from debug, it should be subgraph 1)
    dpu_subgraph = None
    for i, subgraph in enumerate(subgraphs):
        print(f"Subgraph {i}: {subgraph.get_name()}")
        if subgraph.has_attr("device"):
            device = subgraph.get_attr("device")
            print(f"  Device: {device}")
            if isinstance(device, str) and device.upper() == "DPU":
                dpu_subgraph = subgraph
                print(f"  -> Selected as DPU subgraph")
                break
    
    if dpu_subgraph is None:
        print("No DPU subgraph found, trying subgraph 1...")
        if len(subgraphs) > 1:
            dpu_subgraph = subgraphs[1]  # Based on debug output, subgraph 1 is DPU
        else:
            print("ERROR: No suitable subgraph found")
            return False
    
    # Create runner
    try:
        dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
    except Exception as e:
        print(f"Failed to create runner: {e}")
        return False
        
    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()
    
    print("=== Tensor Information ===")
    for i, tensor in enumerate(input_tensors):
        print(f"Input {i}: {tensor.name}, shape: {tensor.dims}, dtype: {tensor.dtype}")
    for i, tensor in enumerate(output_tensors):
        print(f"Output {i}: {tensor.name}, shape: {tensor.dims}, dtype: {tensor.dtype}")
    
    # Test with known working format from debug
    input_shape = tuple(input_tensors[0].dims)  # (1, 416, 416, 3)
    
    print(f"\n=== Testing with int8 format (based on debug) ===")
    try:
        # Create test image - use the format that worked in debug
        # Debug showed: Data range: [0.000, 1.000] with float32, but model expects xint8
        
        # Method 1: Create image data as int8 directly
        input_data = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        
        print(f"Data shape: {input_data.shape}")
        print(f"Data type: {input_data.dtype}")
        print(f"Data range: [{input_data.min()}, {input_data.max()}]")
        
        # Prepare outputs
        output_arrays = []
        for tensor in output_tensors:
            output_arrays.append(np.zeros(tuple(tensor.dims), dtype=np.float32))
        
        # Test inference
        job_id = dpu_runner.execute_async([input_data], output_arrays)
        dpu_runner.wait(job_id)
        
        print(f"✓ SUCCESS with int8")
        
        # Check outputs
        for i, output in enumerate(output_arrays):
            print(f"  Output {i}: shape={output.shape}, range=[{output.min():.6f}, {output.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ FAILED with int8: {e}")
        
        # Try alternative: uint8 converted to proper range
        try:
            print(f"\n=== Testing with uint8 converted to int8 ===")
            # Create realistic image data
            temp_image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
            # Convert to int8 range
            input_data = (temp_image.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            
            print(f"Data shape: {input_data.shape}")
            print(f"Data type: {input_data.dtype}")
            print(f"Data range: [{input_data.min()}, {input_data.max()}]")
            
            job_id = dpu_runner.execute_async([input_data], output_arrays)
            dpu_runner.wait(job_id)
            
            print(f"✓ SUCCESS with uint8->int8 conversion")
            return True
            
        except Exception as e2:
            print(f"✗ FAILED with uint8->int8: {e2}")
            return False

if __name__ == "__main__":
    success = test_yolox_simple()
    if success:
        print("\n=== SUCCESS: Found working data format ===")
    else:
        print("\n=== FAILED: No working data format found ===")
        print("Check if DPU is properly loaded and model is correctly compiled")
