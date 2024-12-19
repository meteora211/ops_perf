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

template<bool exclusive>
__global__ void scanDoubleBuffer(float* lhs, float* partial_sum, float* res, int M);

template<bool exclusive>
__global__ void add_kernel(float* res, const float* partial_sum);

template<bool exclusive>
__global__ void scanBrentKung(float* lhs, float* partial_sum, float* res, int M);


void scanBase(float* lhs, float* res, int M, bool exclusive = true) {
  res[0] = exclusive ? 0.0 : lhs[0];
  for (int i = 1; i < M; ++i) {
    res[i] = res[i - 1] + lhs[i];
  }
}

__global__ void scanNaive(float* lhs, float* partial_sum, float* res, int M) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  res[index] =  lhs[index];
  __syncthreads();
  for (int stride = 1; stride < BLOCK_SIZE; stride *=2) {
    float v;
    if (threadIdx.x < BLOCK_SIZE - stride) {
      v = res[index];
    }
    __syncthreads();
    if (threadIdx.x < BLOCK_SIZE - stride) {
      res[index + stride] += v;
    }
    __syncthreads();
  }
  if (threadIdx.x == BLOCK_SIZE - 1) partial_sum[blockIdx.x] = res[index];
}

__global__ void scanShareMem(float* lhs, float* partial_sum, float* res, int M) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid =threadIdx.x;
  __shared__ float buffer[BLOCK_SIZE];
  if (index < M) buffer[tid] =  lhs[index];
  __syncthreads();
  for (int stride = 1; stride < BLOCK_SIZE; stride *=2) {
    float v;
    if (tid < BLOCK_SIZE - stride) {
      v = buffer[tid];
    }
    __syncthreads();
    if (tid < BLOCK_SIZE - stride) {
      buffer[tid + stride] += v;
    }
    __syncthreads();
  }
  if (threadIdx.x == BLOCK_SIZE - 1) partial_sum[blockIdx.x] = buffer[tid];
  if (index < M) res[index] = buffer[tid];
}

template<>
__global__ void scanDoubleBuffer<true>(float* lhs, float* partial_sum, float* res, int M) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid =threadIdx.x;
  __shared__ float buffer_input[BLOCK_SIZE];
  __shared__ float buffer_output[BLOCK_SIZE];
  float* input_pointer = buffer_input;
  float* output_pointer = buffer_output;
  if (tid == 0) buffer_input[tid] = 0;
  if (index < M && tid > 0) buffer_input[tid] =  lhs[index - 1];
  __syncthreads();
  for (int stride = 1; stride < BLOCK_SIZE; stride *=2) {
    if (tid < stride) {
      output_pointer[tid] = input_pointer[tid];
    } else {
      output_pointer[tid] = input_pointer[tid - stride] + input_pointer[tid];
    }
    __syncthreads();
    // swap buffer pointer
    float* tmp;
    tmp = output_pointer;
    output_pointer = input_pointer;
    input_pointer = tmp;
  }
  // after swap, input_pointer hold the final result
  if (threadIdx.x == BLOCK_SIZE - 1) partial_sum[blockIdx.x] = input_pointer[tid] + lhs[index];
  if (index < M) res[index] = input_pointer[tid];
}

template<>
__global__ void scanDoubleBuffer<false>(float* lhs, float* partial_sum, float* res, int M) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid =threadIdx.x;
  __shared__ float buffer_input[BLOCK_SIZE];
  __shared__ float buffer_output[BLOCK_SIZE];
  float* input_pointer = buffer_input;
  float* output_pointer = buffer_output;
  if (index < M) buffer_input[tid] =  lhs[index];
  __syncthreads();
  for (int stride = 1; stride < BLOCK_SIZE; stride *=2) {
    if (tid < stride) {
      output_pointer[tid] = input_pointer[tid];
    } else {
      output_pointer[tid] = input_pointer[tid - stride] + input_pointer[tid];
    }
    __syncthreads();
    // swap buffer pointer
    float* tmp;
    tmp = output_pointer;
    output_pointer = input_pointer;
    input_pointer = tmp;
  }
  // after swap, input_pointer hold the final result
  if (threadIdx.x == BLOCK_SIZE - 1) partial_sum[blockIdx.x] = input_pointer[tid];
  if (index < M) res[index] = input_pointer[tid];
}

template<>
__global__ void scanBrentKung<true>(float* lhs, float* partial_sum, float* res, int M) {
  unsigned int segment = blockIdx.x * blockDim.x;
  unsigned int tid = threadIdx.x;
  __shared__ float buffer[BLOCK_SIZE];
  if (segment + tid < M) buffer[tid] = lhs[segment + tid];
  __syncthreads();

  // reduction
  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    unsigned int reindex = (tid + 1) * stride * 2 - 1;
    if (reindex < BLOCK_SIZE) {
      buffer[reindex] += buffer[reindex - stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == BLOCK_SIZE - 1) {
    partial_sum[blockIdx.x] = buffer[tid];
    buffer[tid] = 0;
  }

  // post reduction
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    unsigned int reindex = (tid + 1) * stride * 2 - 1;
    if (reindex < BLOCK_SIZE) {
      float tmp = buffer[reindex];
      buffer[reindex] = tmp + buffer[reindex - stride];
      buffer[reindex - stride] = tmp;
    }
    __syncthreads();
  }

  if (segment + tid < M) res[segment + tid] = buffer[tid];
}

template<>
__global__ void scanBrentKung<false>(float* lhs, float* partial_sum, float* res, int M) {
  unsigned int segment = blockIdx.x * blockDim.x;
  unsigned int tid = threadIdx.x;
  __shared__ float buffer[BLOCK_SIZE];
  if (segment + tid < M) buffer[tid] = lhs[segment + tid];
  __syncthreads();

  // reduction
  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    // 1: 1=1+0, 3=3+2, 5=5+4, 7=7+6
    // 2: 3=3+1, 7=7+5
    // 4: 7=7+3
    unsigned int reindex = (tid + 1) * stride * 2 - 1;
    if (reindex < BLOCK_SIZE) {
      buffer[reindex] += buffer[reindex - stride];
    }
    __syncthreads();
  }
  // post reduction
  for (int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) {
    // 2: 5=3+5
    // 1: 2=1+2,4=3+4,6=5+6
    unsigned int reindex = (tid + 1) * stride * 2 - 1;
    if (reindex + stride < BLOCK_SIZE) {
      buffer[reindex + stride] += buffer[reindex];
    }
    __syncthreads();
  }

  if (threadIdx.x == BLOCK_SIZE - 1) partial_sum[blockIdx.x] = buffer[tid];
  if (segment + tid < M) res[segment + tid] = buffer[tid];
}

template<>
__global__ void add_kernel<false>(float* res, const float* partial_sum) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) res[index] += partial_sum[blockIdx.x - 1];
}

template<>
__global__ void add_kernel<true>(float* res, const float* partial_sum) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x > 0) res[index] += partial_sum[blockIdx.x];
}


torch::Tensor scan(const torch::Tensor& input, bool exclusive = false) {
  auto input_size = input.size(0);
  dim3 thread_per_block(BLOCK_SIZE);
  unsigned int numblock = cdiv(input_size, BLOCK_SIZE);
  dim3 block_per_grid(numblock);
  auto res = torch::zeros_like(input);
  auto partial_sum = torch::zeros({numblock}, input.options());

  // scanNaive<<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), partial_sum.data_ptr<float>(), res.data_ptr<float>(), input_size);
  if (exclusive) {
    scanDoubleBuffer<true><<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), partial_sum.data_ptr<float>(), res.data_ptr<float>(), input_size);
  } else {
    // scanDoubleBuffer<false><<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), partial_sum.data_ptr<float>(), res.data_ptr<float>(), input_size);
    scanBrentKung<false><<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), partial_sum.data_ptr<float>(), res.data_ptr<float>(), input_size);
  }


  if (numblock > 1) {
    const auto & sum = scan(partial_sum, exclusive);
    if (exclusive) {
      add_kernel<true><<<block_per_grid, thread_per_block>>>(res.data_ptr<float>(), sum.data_ptr<float>());
    } else {
      add_kernel<false><<<block_per_grid, thread_per_block>>>(res.data_ptr<float>(), sum.data_ptr<float>());
    }
  }

  return res;
}
