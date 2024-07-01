#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHANNELS 3

__global__ void color2grayKernel(unsigned char *Pin, unsigned char *Pout,
                                 int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = Pin[rgbOffset + 0];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];

    Pout[grayOffset] = (unsigned char)(0.21f * r + 0.72f * g + 0.07f * b);
  }
}

// void color2gray(unsigned char *Pin, unsigned char *Pout,
//                                  int width, int height) {
//   unsigned char* Pdin;
//   unsigned char* Pdout;
//
//   uint image_size = width * height;
//   uint image_size_with_channel = image_size * CHANNELS;
//   cudaMalloc(&Pdin, image_size_with_channel);
//   cudaMemcpy(Pdin, Pin, image_size_with_channel, cudaMemcpyHostToDevice);
//   cudaMalloc(&Pdout, image_size);
//
//   color2grayKernel(Pdin, Pdout, width, height);
//   cudaMemcpy(Pout, Pdout, image_size, cudaMemcpyDeviceToHost);
//   cudaFree(Pdin);
//   cudaFree(Pdout);
// }

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}

torch::Tensor color2gray(torch::Tensor image) {
  assert(image.device().type() == torch::kCUDA);
  // assert(image.type() == torch::kByte);

  const auto height = image.size(0);
  const auto width = image.size(1);

  auto result = torch::empty(
      {height, width, 1},
      torch::TensorOptions().dtype(torch::kByte).device(image.device()));

  dim3 thread_per_block(16, 16);
  dim3 block_per_grid(cdiv(width, 16), cdiv(height, 16));

  color2grayKernel<<<block_per_grid, thread_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
    image.data_ptr<unsigned char>(),
    result.data_ptr<unsigned char>(),
    width,
    height
  );

  // check CUDA error status (calls cudaGetLastError())
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return result;
}
