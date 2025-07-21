#!/usr/bin/env python3
"""
Debug script to inspect YOLOX model tensor information
"""

import numpy as np
try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

def debug_model(model_path):
    """Debug model tensor information"""
    
    print(f"=== Debugging model: {model_path} ===")
    
    try:
        # Load xmodel
        graph = xir.Graph.deserialize(model_path)
        root_subgraph = graph.get_root_subgraph()
        
        print(f"Root subgraph name: {root_subgraph.get_name()}")
        print(f"Root subgraph device: {root_subgraph.get_attr('device') if root_subgraph.has_attr('device') else 'Unknown'}")
        
        # Get subgraphs
        subgraphs = []
        try:
            if hasattr(root_subgraph, 'children_topological_sort'):
                subgraphs = root_subgraph.children_topological_sort()
            elif hasattr(root_subgraph, 'get_children'):
                subgraphs = root_subgraph.get_children()
            elif hasattr(root_subgraph, 'children'):
                subgraphs = root_subgraph.children
            else:
                subgraphs = [root_subgraph]
        except AttributeError:
            subgraphs = [root_subgraph]
        
        print(f"Number of subgraphs: {len(subgraphs)}")
        
        # Find DPU subgraph
        dpu_subgraph = None
        for i, subgraph in enumerate(subgraphs):
            print(f"Subgraph {i}: {subgraph.get_name()}")
            if subgraph.has_attr("device"):
                device = subgraph.get_attr("device")
                print(f"  Device: {device}")
                if isinstance(device, str) and device.upper() == "DPU":
                    dpu_subgraph = subgraph
                    print(f"  -> Selected as DPU subgraph")
        
        if dpu_subgraph is None:
            print("No DPU subgraph found, using first subgraph")
            dpu_subgraph = subgraphs[0] if subgraphs else root_subgraph
        
        # Create DPU runner
        print(f"\n=== Creating DPU runner ===")
        dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
        
        # Get tensor information
        input_tensors = dpu_runner.get_input_tensors()
        output_tensors = dpu_runner.get_output_tensors()
        
        print(f"\n=== Input Tensors ({len(input_tensors)}) ===")
        for i, tensor in enumerate(input_tensors):
            print(f"Input {i}:")
            print(f"  Name: {tensor.name}")
            print(f"  Shape: {tensor.dims}")
            print(f"  Data type: {tensor.dtype}")
            if hasattr(tensor, 'get_data_type'):
                print(f"  Get data type: {tensor.get_data_type()}")
            print(f"  Total elements: {np.prod(tensor.dims)}")
        
        print(f"\n=== Output Tensors ({len(output_tensors)}) ===")
        for i, tensor in enumerate(output_tensors):
            print(f"Output {i}:")
            print(f"  Name: {tensor.name}")
            print(f"  Shape: {tensor.dims}")
            print(f"  Data type: {tensor.dtype}")
            if hasattr(tensor, 'get_data_type'):
                print(f"  Get data type: {tensor.get_data_type()}")
            print(f"  Total elements: {np.prod(tensor.dims)}")
        
        # Test with dummy input
        print(f"\n=== Testing with dummy input ===")
        input_arrays = []
        output_arrays = []
        
        for i, tensor in enumerate(input_tensors):
            shape = tuple(tensor.dims)
            print(f"Creating input array {i} with shape {shape}")
            
            # Create dummy input data
            if i == 0:  # Main input tensor (image)
                # Try different data ranges
                dummy_input = np.random.rand(*shape).astype(np.float32)
                # dummy_input = np.ones(shape, dtype=np.float32) * 0.5  # Alternative
            else:
                dummy_input = np.zeros(shape, dtype=np.float32)
            
            print(f"  Data type: {dummy_input.dtype}")
            print(f"  Data range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
            input_arrays.append(dummy_input)
        
        for i, tensor in enumerate(output_tensors):
            shape = tuple(tensor.dims)
            print(f"Creating output array {i} with shape {shape}")
            output_array = np.zeros(shape, dtype=np.float32)
            output_arrays.append(output_array)
        
        try:
            print("Executing DPU inference...")
            job_id = dpu_runner.execute_async(input_arrays, output_arrays)
            dpu_runner.wait(job_id)
            print("✓ DPU inference successful!")
            
            # Check outputs
            for i, output in enumerate(output_arrays):
                print(f"Output {i} shape: {output.shape}")
                print(f"Output {i} range: [{output.min():.6f}, {output.max():.6f}]")
                print(f"Output {i} mean: {output.mean():.6f}")
                
        except Exception as e:
            print(f"✗ DPU inference failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Model debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    model_path = "yolox_nano_pt.xmodel"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    debug_model(model_path)
