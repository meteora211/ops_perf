#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

constexpr int KERNELSIZE = 3;
__global__ void blurKernel(float* Pin, float* Pout, int width, int height) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int pad = (KERNELSIZE + 1) / 2;
  float value = 0.0;
  int cnt = 0;
  if (col < width && row < height) {
    for (int i = -pad; i < KERNELSIZE - pad; ++i) {
      if (row + i < 0 || row + i >= height) continue;
      for (int j = -pad; j < KERNELSIZE - pad; ++j) {
        if (col + j < 0 || col + j >= width) continue;
        value += Pin[(row + i) * width + (col + j)];
        ++cnt;
      }
    }
    Pout[row * width + col] = value / cnt;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}

torch::Tensor blur(torch::Torch image) {
  assert(image.device().type() == torch::kCUDA);

  const auto height = image.size(0);
  const auto width = image.size(1);
  const auto channels = image.size(2);

  auto result = torch::empty_like(image);

  dim3 thread_per_block(16, 16);
  dim3 block_per_grid(cdiv(width, 16), cdiv(height, 16));
}
