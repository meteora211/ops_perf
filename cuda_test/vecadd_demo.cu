#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>

#include "utils.h"

#define BLOCK_SIZE 128

void vecAddRef(float* lhs, float* rhs, float* res, int N) {
  for (int i = 0; i < N; ++i) {
    res[i] = lhs[i] + rhs[i];
  }
}

__global__ void vecAddKernel(float* lhs, float* rhs, float* res, int N) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < N) res[i] = lhs[i] + rhs[i];
}

void vecAddCU(float* lhs, float* rhs, float* res, int N) {
  float* d_lhs;
  float* d_rhs;
  float* d_res;
  dim3 threadPerBlock(BLOCK_SIZE);
  dim3 blockPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  cudaMalloc(&d_lhs, N * sizeof(float));
  cudaMemcpy(d_lhs, lhs, N, cudaMemcpyHostToDevice);
  cudaMalloc(&d_rhs, N * sizeof(float));
  cudaMemcpy(d_rhs, rhs, N, cudaMemcpyHostToDevice);

  cudaMalloc(&d_res, N * sizeof(float));

  vecAddKernel<<<blockPerGrid,threadPerBlock>>>(d_lhs, d_rhs, d_res, N);

  cudaMemcpy(res, d_res, N, cudaMemcpyDeviceToHost);
  
  cudaFree(d_lhs);
  cudaFree(d_rhs);
  cudaFree(d_res);
}

int main() {
  int N = 4096;
  float* hA = static_cast<float*>(malloc(N * sizeof(float)));
  float* hB = static_cast<float*>(malloc(N * sizeof(float)));
  float* hRes = static_cast<float*>(malloc(N * sizeof(float)));
  fullfill_rand(hA, N);
  fullfill_rand(hB, N);

  vecAddCU(hA, hB, hRes, N);
  
  float* hRef = static_cast<float*>(malloc(N * sizeof(float)));
  vecAddRef(hA, hB, hRef, N);

  matrixChecker(hRes, hRef, 1, 1);
}
