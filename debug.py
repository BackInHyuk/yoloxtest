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
    subgraphs = []
    
    root_subgraph = graph.get_root_subgraph()
    try:
        if hasattr(root_subgraph, 'children_topological_sort'):
            subgraphs = root_subgraph.children_topological_sort()
        else:
            subgraphs = [root_subgraph]
    except:
        subgraphs = [root_subgraph]
    
    # Find DPU subgraph
    dpu_subgraph = None
    for subgraph in subgraphs:
        if subgraph.has_attr("device"):
            device = subgraph.get_attr("device")
            if isinstance(device, str) and device.upper() == "DPU":
                dpu_subgraph = subgraph
                break
    
    if dpu_subgraph is None:
        dpu_subgraph = subgraphs[0]
    
    # Create runner
    dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()
    
    print("=== Tensor Information ===")
    for i, tensor in enumerate(input_tensors):
        print(f"Input {i}: {tensor.name}, shape: {tensor.dims}, dtype: {tensor.dtype}")
    
    # Test different data types
    input_shape = tuple(input_tensors[0].dims)  # (1, 416, 416, 3)
    
    test_cases = [
        ("uint8 [0,255]", lambda: np.random.randint(0, 256, input_shape, dtype=np.uint8)),
        ("int8 [-128,127]", lambda: np.random.randint(-128, 128, input_shape, dtype=np.int8)),
        ("float32 [0,1]", lambda: np.random.rand(*input_shape).astype(np.float32)),
        ("float32 [0,255]", lambda: (np.random.rand(*input_shape) * 255).astype(np.float32)),
        ("real image uint8", lambda: create_real_image_uint8(input_shape)),
        ("real image int8", lambda: create_real_image_int8(input_shape)),
    ]
    
    for test_name, data_generator in test_cases:
        print(f"\n=== Testing: {test_name} ===")
        try:
            # Generate test data
            input_data = data_generator()
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
            
            print(f"✓ SUCCESS: {test_name}")
            
            # Check outputs
            for i, output in enumerate(output_arrays):
                print(f"  Output {i}: shape={output.shape}, range=[{output.min():.6f}, {output.max():.6f}]")
            
            return True  # Found working data type
            
        except Exception as e:
            print(f"✗ FAILED: {test_name} - {e}")
            continue
    
    return False

def create_real_image_uint8(shape):
    """Create a real-looking image in uint8 format"""
    # Create a synthetic image
    _, h, w, c = shape
    image = np.zeros((h, w, c), dtype=np.uint8)
    
    # Add some patterns
    image[:, :, 0] = 128  # Red channel
    image[:, :, 1] = 64   # Green channel  
    image[:, :, 2] = 192  # Blue channel
    
    # Add some noise
    noise = np.random.randint(0, 50, (h, w, c), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    
    return np.expand_dims(image, axis=0)

def create_real_image_int8(shape):
    """Create a real-looking image in int8 format"""
    uint8_image = create_real_image_uint8(shape)
    # Convert to int8 by subtracting 128
    return (uint8_image.astype(np.int16) - 128).astype(np.int8)

if __name__ == "__main__":
    success = test_yolox_simple()
    if success:
        print("\n=== SUCCESS: Found working data format ===")
    else:
        print("\n=== FAILED: No working data format found ===")
