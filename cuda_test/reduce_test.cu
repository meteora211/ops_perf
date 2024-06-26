#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>

#include "utils.h"

#define BLOCK_SIZE 128

void reduceBase(float* lhs, float* res, int M) {
  float sum = 0;
  for (int i = 0; i < M; ++i) {
    sum += lhs[i];
  }
  res[0] = sum;
}

__global__ void reduceNaive(float* lhs, float* res, int M) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float cache[BLOCK_SIZE];
  cache[threadIdx.x] = lhs[i];
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (stride * 2) == 0) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  
  if (tid == 0) res[blockIdx.x] = cache[0];
}

__global__ void reduceSeq(float* lhs, float* res, int M) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  
  __shared__ float cache[BLOCK_SIZE];
  cache[tid] = lhs[i];
  __syncthreads();
  for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) cache[tid] += cache[tid + stride];
    __syncthreads();
  }
  
  if(tid == 0) res[blockIdx.x] = cache[0];
}


int main() {
  std::cout << "test reduce naive" << std::endl;
  int M = 4096;
  const bool checkResult = true;
  const int iteration = 50;
  int blockNum = M / BLOCK_SIZE;
  
  float* hA = static_cast<float*>(malloc(M * sizeof(float)));
  float* hB = static_cast<float*>(malloc(blockNum * sizeof(float)));
  fullfill_rand(hA, M);
  
  float* dA;
  cudaMalloc(&dA, M * sizeof(float));
  float* dB;
  cudaMalloc(&dB, blockNum * sizeof(float));
  
  cudaMemcpy(dA, hA, M * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 threadPerBlock(BLOCK_SIZE);
  dim3 blockPerGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // warm up
  for (int i = 0; i < 20; ++i) {
    reduceNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  }
  {
    CUProfiler profiler("naive");
    for (int i = 0; i < iteration; ++i) reduceNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  }
  {
    CUProfiler profiler("sequential");
    for (int i = 0; i < iteration; ++i) reduceSeq<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  }

  reduceSeq<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  cudaMemcpy(hB, dB, blockNum * sizeof(float), cudaMemcpyDeviceToHost);
  if (checkResult) {
    float* hCuda = static_cast<float*>(malloc(sizeof(float)));
    reduceBase(hB, hCuda, blockNum);

    float* hRef = static_cast<float*>(malloc(sizeof(float)));
    reduceBase(hA, hRef, M);

    matrixChecker(hCuda, hRef, 1, 1);
  }
}