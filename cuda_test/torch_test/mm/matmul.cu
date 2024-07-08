#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matmulNaiveKernel(float* M, float* N, float* P, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0.0;
  if (row < width && col < width) {
    for (int k = 0; k < width; ++k) {
      value += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = value;
  }
}

__global__ void matmulSharedKernel(float* M, float* N, float* P, int width) {
  __shared__ float tileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileN[TILE_WIDTH][TILE_WIDTH];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0.0;
  if (row < width && col < width) {
    for (int k = 0; k < width; k += TILE_WIDTH) {

      for (int inner = 0; inner < TILE_WIDTH; ++inner) {
        value = tileM[][inner] * tileN[inner][];

      }

    }
    P[row * width + col] = value;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}

torch::Tensor matmul(torch::Tensor lhs, torch::Tensor rhs) {
  assert(input.device().type() == torch::kCUDA);

  const auto width = lhs.size(0);

  auto result = torch::empty_like(image);

  dim3 thread_per_block(16, 16);
  dim3 block_per_grid(cdiv(width, 16), cdiv(height, 16));

  matmulKernel<<<block_per_grid, thread_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    rhs.data_ptr<float>(),
    lhs.data_ptr<float>(),
    result.data_ptr<float>(),
    width,
  );

  // check CUDA error status (calls cudaGetLastError())
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return result;
}
