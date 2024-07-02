#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void blurKernel(unsigned char* Pin, unsigned char* Pout, int width, int height, int kernel_size) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int channel = threadIdx.z;

  int pad = (kernel_size + 1) / 2;
  float value = 0.0;
  int cnt = 0;
  int offset = width * height * channel;
  if (col < width && row < height) {
    for (int i = -pad; i < kernel_size - pad; ++i) {
      if (row + i < 0 || row + i >= height) continue;
      for (int j = -pad; j < kernel_size - pad; ++j) {
        if (col + j < 0 || col + j >= width) continue;
        value += Pin[offset + (row + i) * width + (col + j)];
        ++cnt;
      }
    }
    Pout[offset + row * width + col] = value / cnt;
  }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}

torch::Tensor blur(torch::Tensor image, int kernel_size) {
  assert(image.device().type() == torch::kCUDA);

  const auto channels = image.size(0);
  const auto height = image.size(1);
  const auto width = image.size(2);

  auto result = torch::empty_like(image);

  dim3 thread_per_block(16, 16, 3);
  dim3 block_per_grid(cdiv(width, 16), cdiv(height, 16));

  blurKernel<<<block_per_grid, thread_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    image.data_ptr<unsigned char>(),
    result.data_ptr<unsigned char>(),
    width,
    height,
    kernel_size
  );
  return result;
}
