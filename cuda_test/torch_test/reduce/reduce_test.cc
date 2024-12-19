#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

constexpr int BLOCK_SIZE = 128;

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}


void reduceBase(float* lhs, float* res, int M) {
  float sum = 0;
  for (int i = 0; i < M; ++i) {
    sum += lhs[i];
  }
  res[0] = sum;
}

__global__ void reduceNaive(float* lhs, float* res, int M) {
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    if (threadIdx.x % stride == 0) lhs[index] += lhs[index + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) res[blockIdx.x] = lhs[index];
}

__global__ void reduceShared(const float* lhs, float* res, int M) {
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  __shared__ float cache[BLOCK_SIZE];
  int stride = BLOCK_SIZE;
  int tid = threadIdx.x;
  float temp = (index < M) ? lhs[index] : 0;
  if (index + stride < M) {
    cache[tid] = temp + lhs[index + stride];
  } else {
    cache[tid] = temp;
  }
  __syncthreads();
  for (stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  if (tid == 0) res[blockIdx.x] = cache[0];
}




torch::Tensor reduce(const torch::Tensor& input) {
  auto input_size = input.size(0);
  dim3 thread_per_block(BLOCK_SIZE);
  constexpr int elem_per_block = BLOCK_SIZE * 2;
  dim3 block_per_grid(cdiv(input_size, elem_per_block));
  auto block_output = torch::zeros({cdiv(input_size, elem_per_block)}, input.options());
  auto res = torch::zeros({1}, input.options());
  reduceShared<<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), block_output.data_ptr<float>(), input_size);
  unsigned int block_output_size = block_output.size(0);

  std::cout << "--------------------BEFORE LOOP--------------------" << block_output_size << std::endl;
  while(block_output_size > elem_per_block) {
    std::cout << "--------------------LOOP--------------------" << block_output_size << std::endl;
    auto temp = torch::zeros({cdiv(block_output_size, elem_per_block)}, input.options());
    dim3 block_per_grid(cdiv(block_output_size, elem_per_block));
    reduceShared<<<block_per_grid, thread_per_block>>>(block_output.data_ptr<float>(), temp.data_ptr<float>(), block_output_size);
    block_output = temp;
    block_output_size = block_output.size(0);
  }
  reduceShared<<<1, thread_per_block>>>(block_output.data_ptr<float>(), res.data_ptr<float>(), block_output_size);

  return res;
}
