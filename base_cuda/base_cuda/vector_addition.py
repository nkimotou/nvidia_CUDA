import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA kernel code
cuda_code = """
__global__ void add_vectors(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Compile the CUDA code
mod = SourceModule(cuda_code)

# Get the kernel function
add_vectors = mod.get_function("add_vectors")

# Define vector size
n = 1000

# Generate random input vectors
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)

# Allocate memory on the device
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(b.nbytes)

# Copy data to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Define block and grid dimensions
block_size = 256
grid_size = (n + block_size - 1) // block_size

# Launch the kernel
add_vectors(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

# Allocate memory on the host for the result
c = np.empty_like(a)

# Copy the result back to the host
cuda.memcpy_dtoh(c, c_gpu)

# Print the result
print("Result of vector addition:", c)
