import torch
import time

# Define the size of the matrices
MATRIX_SIZE = 30000

print(f"--- GPU Matrix Multiplication Test ---")
print(f"Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")

# --- GPU Test ---
# Check if a ROCm-compatible GPU is available
if torch.cuda.is_available():
    print(f"\nFound compatible GPU: {torch.cuda.get_device_name(0)}")
    
    # Set the device to your GPU
    device = torch.device("cuda")

    # Create two random matrices directly on the GPU
    matrix_a_gpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    matrix_b_gpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

    # --- Warm-up run ---
    # The first GPU operation can have some startup overhead.
    # We do one multiplication first to warm up the GPU.
    print("Performing a warm-up multiplication...")
    _ = torch.matmul(matrix_a_gpu, matrix_b_gpu)
    torch.cuda.synchronize() # Wait for the warm-up to finish

    # --- Timed run ---
    print("Starting timed GPU multiplication...")
    start_time_gpu = time.time()

    # Perform the matrix multiplication on the GPU
    result_gpu = torch.matmul(matrix_a_gpu, matrix_b_gpu)

    # Wait for the GPU to finish the computation before stopping the timer
    torch.cuda.synchronize()
    
    end_time_gpu = time.time()
    gpu_duration = end_time_gpu - start_time_gpu

    print(f"✅ GPU multiplication finished in: {gpu_duration:.4f} seconds")

else:
    print("\n⚠️ No ROCm-compatible GPU found. Skipping GPU test.")


# --- CPU Test (for comparison) ---
print("\n--- CPU Matrix Multiplication Test ---")

# Create two random matrices on the CPU
matrix_a_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE)
matrix_b_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE)

# --- Timed run ---
print("Starting timed CPU multiplication...")
start_time_cpu = time.time()

# Perform the matrix multiplication on the CPU
result_cpu = torch.matmul(matrix_a_cpu, matrix_b_cpu)

end_time_cpu = time.time()
cpu_duration = end_time_cpu - start_time_cpu

print(f"✅ CPU multiplication finished in: {cpu_duration:.4f} seconds")

# --- Comparison ---
if torch.cuda.is_available():
    speedup = cpu_duration / gpu_duration
    print(f"\n--- Results ---")
    print(f"GPU was approximately {speedup:.2f}x faster than the CPU for this task.")