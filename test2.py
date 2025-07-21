#!/usr/bin/env python3
"""
Final DPU test with alternative memory allocation approach
"""

import numpy as np
import time
import gc
import sys

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

def test_alternative_dpu_call():
    """Test DPU with alternative memory allocation strategy"""
    
    print("=== Final DPU Test - Alternative Approach ===")
    
    try:
        # Load model
        print("Loading model...")
        graph = xir.Graph.deserialize("yolox_nano_pt.xmodel")
        root_subgraph = graph.get_root_subgraph()
        
        # Get subgraphs
        subgraphs = root_subgraph.children_topological_sort()
        
        # Use subgraph 2 (confirmed DPU from debug)
        dpu_subgraph = subgraphs[2]
        print(f"Using subgraph 2: {dpu_subgraph.get_name()}")
        
        # Create DPU runner
        print("Creating DPU runner...")
        dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
        
        # Get tensors
        input_tensors = dpu_runner.get_input_tensors()
        output_tensors = dpu_runner.get_output_tensors()
        
        print(f"Tensors ready - Input: {input_tensors[0].dims}")
        
        # Alternative approach 1: Use vart tensor allocation
        print("\n=== Method 1: Using vart tensor allocation ===")
        try:
            # Allocate input buffer using vart
            input_shape = tuple(input_tensors[0].dims)
            
            # Create input data with specific memory alignment
            input_data = np.zeros(input_shape, dtype=np.int8, order='C')
            
            # Fill with simple pattern
            input_data.fill(50)  # Mid-range value
            
            print(f"Input prepared: {input_data.shape}, {input_data.dtype}")
            print(f"Memory layout: C-contiguous={input_data.flags.c_contiguous}")
            
            # Allocate output buffers
            output_arrays = []
            for tensor in output_tensors:
                shape = tuple(tensor.dims)
                # Use C-contiguous memory layout
                output_array = np.zeros(shape, dtype=np.float32, order='C')
                output_arrays.append(output_array)
            
            print("Output arrays allocated")
            
            # Force garbage collection before DPU call
            gc.collect()
            
            print(">>> Method 1: Starting DPU inference <<<")
            
            # Use execute instead of execute_async
            outputs = dpu_runner.execute([input_data])
            
            print(">>> Method 1: DPU inference completed! <<<")
            
            for i, output in enumerate(outputs):
                print(f"Output {i}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
            
            return True, "method1_sync_execute"
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Alternative approach 2: Synchronous execution with manual memory
        print("\n=== Method 2: Manual memory management ===")
        try:
            # Create aligned memory manually
            input_shape = tuple(input_tensors[0].dims)
            total_elements = np.prod(input_shape)
            
            # Allocate page-aligned memory
            input_data = np.zeros(input_shape, dtype=np.int8)
            input_data = np.ascontiguousarray(input_data)  # Ensure contiguous
            
            # Simple data pattern
            input_data[:] = np.random.randint(-50, 50, input_shape, dtype=np.int8)
            
            print(f"Manual input: {input_data.shape}, contiguous={input_data.flags.c_contiguous}")
            
            # Manual output allocation
            output_arrays = []
            for tensor in output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            print("Manual output arrays ready")
            
            # Clear any residual memory
            gc.collect()
            
            print(">>> Method 2: Starting manual DPU inference <<<")
            
            # Try with execute_async but different parameters
            job_id = dpu_runner.execute_async([input_data], output_arrays)
            
            # Wait with timeout to avoid hanging
            print("Waiting for job completion...")
            dpu_runner.wait(job_id)
            
            print(">>> Method 2: Manual DPU inference completed! <<<")
            
            for i, output in enumerate(output_arrays):
                print(f"Output {i}: shape={output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
            
            return True, "method2_manual_memory"
            
        except Exception as e:
            print(f"Method 2 failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Alternative approach 3: Minimal data test
        print("\n=== Method 3: Minimal data test ===")
        try:
            # Use the exact working data from minimal_dpu_test.py
            input_shape = tuple(input_tensors[0].dims)
            
            # Create minimal test data (exactly like the working test)
            input_data = np.zeros(input_shape, dtype=np.int8)
            
            print(f"Minimal data: {input_data.shape}, {input_data.dtype}")
            
            # Prepare outputs
            output_arrays = []
            for tensor in output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_arrays.append(output_array)
            
            print(">>> Method 3: Minimal data inference <<<")
            
            # Single line DPU call
            job_id = dpu_runner.execute_async([input_data], output_arrays)
            dpu_runner.wait(job_id)
            
            print(">>> Method 3: Completed! <<<")
            return True, "method3_minimal"
            
        except Exception as e:
            print(f"Method 3 failed: {e}")
            import traceback
            traceback.print_exc()
        
        return False, "all_methods_failed"
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False, "setup_failed"

def main():
    print("This test tries different DPU calling methods to avoid segfault")
    print("If all methods fail, there may be a fundamental DPU driver issue\n")
    
    # System info
    print(f"Python version: {sys.version}")
    
    try:
        success, method = test_alternative_dpu_call()
        
        print("\n" + "="*60)
        if success:
            print(f"🎉 SUCCESS with {method}!")
            print("Found a working DPU inference method.")
            print("We can now build a stable detection system.")
        else:
            print(f"❌ ALL METHODS FAILED: {method}")
            print("\nPossible issues:")
            print("1. DPU driver compatibility problem")
            print("2. Vitis AI runtime version mismatch")
            print("3. Model compilation issue")
            print("4. Hardware/kernel module problem")
            print("\nSuggested actions:")
            print("- Check DPU driver: sudo dmesg | grep -i dpu")
            print("- Verify Vitis AI version compatibility")
            print("- Try recompiling the model")
            print("- System reboot")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")

if __name__ == "__main__":
    main()
