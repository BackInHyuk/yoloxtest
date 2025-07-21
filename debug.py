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
        
        if root_subgraph is None:
            print("ERROR: Failed to get root subgraph")
            return False, "no_root_subgraph"
        
        print(f"Root subgraph: {root_subgraph.get_name()}")
        
        # Get subgraphs with safe error handling
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
        except Exception as e:
            print(f"Error getting subgraphs: {e}")
            subgraphs = [root_subgraph]
        
        if not subgraphs:
            print("ERROR: No subgraphs found")
            return False, "no_subgraphs"
        
        print(f"Found {len(subgraphs)} subgraphs")
        
        # Find DPU subgraph with safe checks
        dpu_subgraph = None
        for i, sg in enumerate(subgraphs):
            if sg is None:
                print(f"Subgraph {i}: None (skipping)")
                continue
                
            try:
                sg_name = sg.get_name() if sg else "Unknown"
                print(f"Subgraph {i}: {sg_name}")
                
                if sg.has_attr("device"):
                    device = sg.get_attr("device")
                    print(f"  Device: {device}")
                    if isinstance(device, str) and device.upper() == "DPU":
                        dpu_subgraph = sg
                        print(f"  -> Selected as DPU subgraph")
                        break
                else:
                    print(f"  No device attribute")
                    
            except Exception as e:
                print(f"  Error checking subgraph {i}: {e}")
                continue
        
        # Fallback selection with safety checks
        if dpu_subgraph is None:
            print("No DPU subgraph found by device attribute")
            if len(subgraphs) > 1 and subgraphs[1] is not None:
                dpu_subgraph = subgraphs[1]
                print("Using subgraph 1 as fallback")
            elif len(subgraphs) > 0 and subgraphs[0] is not None:
                dpu_subgraph = subgraphs[0]
                print("Using subgraph 0 as fallback")
            else:
                print("ERROR: No valid subgraph available")
                return False, "no_valid_subgraph"
        
        if dpu_subgraph is None:
            print("ERROR: Selected DPU subgraph is None")
            return False, "dpu_subgraph_none"
        
        try:
            selected_name = dpu_subgraph.get_name()
            print(f"Selected DPU subgraph: {selected_name}")
            
            # Check if it's the problematic root subgraph
            if selected_name == "root":
                print("WARNING: Selected subgraph is 'root' which may not be compiled")
                # Try to find alternative
                for i, sg in enumerate(subgraphs):
                    if sg is not None and sg != dpu_subgraph:
                        try:
                            alt_name = sg.get_name()
                            if alt_name != "root":
                                dpu_subgraph = sg
                                print(f"Switching to alternative subgraph: {alt_name}")
                                break
                        except:
                            continue
        except Exception as e:
            print(f"Error getting subgraph name: {e}")
            return False, "subgraph_name_error"
        
        # Create runner with error handling
        print("2. Creating DPU runner...")
        try:
            dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            if dpu_runner is None:
                print("ERROR: Failed to create DPU runner")
                return False, "runner_creation_failed"
        except Exception as e:
            print(f"ERROR: DPU runner creation failed: {e}")
            return False, f"runner_error: {e}"
        
        # Get tensors with error handling
        print("3. Getting tensor information...")
        try:
            input_tensors = dpu_runner.get_input_tensors()
            output_tensors = dpu_runner.get_output_tensors()
            
            if not input_tensors or not output_tensors:
                print("ERROR: Failed to get tensor information")
                return False, "tensor_info_failed"
                
        except Exception as e:
            print(f"ERROR: Getting tensors failed: {e}")
            return False, f"tensor_error: {e}"
        
        print(f"Input tensors: {len(input_tensors)}")
        print(f"Output tensors: {len(output_tensors)}")
        
        # Print tensor details safely
        for i, tensor in enumerate(input_tensors):
            try:
                print(f"Input {i}: shape={tensor.dims}, dtype={tensor.dtype}")
            except Exception as e:
                print(f"Input {i}: Error getting info - {e}")
        
        for i, tensor in enumerate(output_tensors):
            try:
                print(f"Output {i}: shape={tensor.dims}, dtype={tensor.dtype}")
            except Exception as e:
                print(f"Output {i}: Error getting info - {e}")
        
        # Test different input data types
        try:
            input_shape = tuple(input_tensors[0].dims)
        except Exception as e:
            print(f"ERROR: Cannot get input shape: {e}")
            return False, "input_shape_error"
        
        test_cases = [
            ("zeros_int8", lambda: np.zeros(input_shape, dtype=np.int8)),
            ("random_int8", lambda: np.random.randint(-128, 127, input_shape, dtype=np.int8)),
        ]
        
        for test_name, data_gen in test_cases:
            print(f"\n4. Testing with {test_name}...")
            
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
                    try:
                        shape = tuple(tensor.dims)
                        output_array = np.zeros(shape, dtype=np.float32)
                        output_arrays.append(output_array)
                        print(f"     Output {j}: shape={shape}, dtype={output_array.dtype}")
                    except Exception as e:
                        print(f"     Error preparing output {j}: {e}")
                        return False, f"output_prep_error_{j}"
                
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
        
        return False, "all_tests_failed"
        
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
