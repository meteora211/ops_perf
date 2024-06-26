// #include "gpu_reduce.h"
#include "utils.h"
#include "baseline.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

double reduce_cuda_executor(void(*cuda_func)(float *, float *, const int),
                   std::shared_ptr<float[]> lhs,
                   std::shared_ptr<float[]> res,
                   int N,
                   dim3 grid,
                   dim3 block) {
  size_t lhs_size = N * sizeof(float);
  // reduce 返回numBlocks个数，后续可以在cpu或者重新使用1个block reduce返回值
  size_t res_size = block.x * sizeof(float);

  float* lhs_device, *res_device;
  cudaMalloc(&lhs_device, lhs_size);
  cudaMalloc(&res_device, res_size);

  cudaMemcpy(lhs_device, lhs.get(), lhs_size, cudaMemcpyHostToDevice);
  cudaMemcpy(res_device, res.get(), res_size, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  cuda_func<<<grid, block>>>(lhs_device, res_device, N);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec;
  cudaEventElapsedTime(&msec, start, end);

  cudaMemcpy(res.get(), res_device, res_size, cudaMemcpyDeviceToHost);


  cudaFree(lhs_device);
  cudaFree(res_device);
  return msec;
}

template<int THREAD_PER_BLOCK>
__global__ void reduce_naive(float* lhs, float* res, int N) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = lhs[i];
    __syncthreads();
    
    // reduce sdata
    for (int i = 1; i < blockDim.x; i*=2) {
        // int stride = i;
        if (tid % (i * 2) == 0) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    
    if (tid==0) res[blockIdx.x] = sdata[0];
}

template<int THREAD_PER_BLOCK>
__global__ void reduce_sequential(float* lhs, float* res, int N) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    sdata[tid] = lhs[i];
    __syncthreads();
    
    // reduce sdata
    for (int i = blockDim.x / 2; i > 0; i=i>>1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    
    if (tid==0) res[blockIdx.x] = sdata[0];
}

double reduce_cuda_naive(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> res, int N) {
  const int BN = 32;

  dim3 grid((N + BN - 1) / BN);
  dim3 block(BN);
  auto tmp_res = std::shared_ptr<float[]>(new float[BN]);
  double dur = reduce_cuda_executor(reduce_naive<BN>, lhs, tmp_res, N, grid, block);
  res[0] = 0;
  for (int i = 0; i < BN; ++i) {
    res[0] += tmp_res[i];
  }
  return dur;
}

double reduce_cuda_sequential(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> res, int N) {
  const int BN = 32;

  dim3 grid((N + BN - 1) / BN);
  dim3 block(BN);
  auto tmp_res = std::shared_ptr<float[]>(new float[BN]);
  double dur = reduce_cuda_executor(reduce_sequential<BN>, lhs, tmp_res, N, grid, block);
  res[0] = 0;
  for (int i = 0; i < BN; ++i) {
    res[0] += tmp_res[i];
  }
  return dur;
}
