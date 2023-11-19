#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>

#include "utils.h"

#define BLOCK_SIZE 16

void matmulBase(float* lhs, float* rhs, float* res, int M, int N, int K) {
  // lhs(M*K) * rhs(K*N) = res(M*N)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        // sum += lhs[i, k] * rhs[k, j];
        sum += lhs[i * K + k] * rhs[k * N + j];
      }
      res[i * N + j] = sum;
    }
  }
}

__global__ void matmulNaive(float* A, float* B, float* C, int M, int N, int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < M && j < N) {
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
  }
}

__global__ void matmulShared(float* A, float* B, float* C, int M, int N, int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = threadIdx.y;
  int bj = threadIdx.x;
  
  float sum = 0;
  __shared__ float shareA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shareB[BLOCK_SIZE][BLOCK_SIZE];
  for (int k = 0; k < K; k+=BLOCK_SIZE) {
    if ((i < M) && ((k + bj) < K)) {
      shareA[bi][bj] = A[i * K + (k + bj)];
    } else {
      shareA[bi][bj] = 0;
    }
    if ((j < N) && ((k + bi) < K)) {
      shareB[bj][bi] = B[(k + bi) * N + j];
    } else {
      shareB[bj][bi] = 0;
    }
    __syncthreads();
    for (int inner = 0; inner < BLOCK_SIZE; ++inner) {
      sum += shareA[bi][inner] * shareB[bj][inner];
    }
    __syncthreads();
  }
  if (i < M && j < N) {
    C[i * N + j] = sum;
  }
}

__global__ void matmulBankOpt(float* A, float* B, float* C, int M, int N, int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = threadIdx.y;
  int bj = threadIdx.x;
  
  float sum = 0;
  __shared__ float shareA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shareB[BLOCK_SIZE][BLOCK_SIZE + 1];
  for (int k = 0; k < K; k+=BLOCK_SIZE) {
    if ((i < M) && ((k + bj) < K)) {
      shareA[bi][bj] = A[i * K + (k + bj)];
    } else {
      shareA[bi][bj] = 0;
    }
    if ((j < N) && ((k + bi) < K)) {
      shareB[bi][bj] = B[(k + bi) * N + j];
    } else {
      shareB[bi][bj] = 0;
    }
    __syncthreads();
    for (int inner = 0; inner < BLOCK_SIZE; ++inner) {
      sum += shareA[bi][inner] * shareB[inner][bj];
    }
    __syncthreads();
  }
  if (i < M && j < N) {
    C[i * N + j] = sum;
  }
}

int main() {
  std::cout << "test matmul naive" << std::endl;
  int M = 1024;
  int N = 3000;
  int K = 2000;
  const bool checkResult = false;
  const int iteration = 50;
  
  float* hA = static_cast<float*>(malloc(M * K * sizeof(float)));
  float* hB = static_cast<float*>(malloc(K * N * sizeof(float)));
  float* hC = static_cast<float*>(malloc(M * N * sizeof(float)));
  fullfill_rand(hA, M * K);
  fullfill_rand(hB, K * N);
  
  float* dA;
  cudaMalloc(&dA, M * K * sizeof(float));
  float* dB;
  cudaMalloc(&dB, K * N * sizeof(float));
  float* dC;
  cudaMalloc(&dC, M * N * sizeof(float));
  
  cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blockPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // warm up
  for (int i = 0; i < 20; ++i) {
    matmulNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
  }
  {
    CUProfiler profiler("naive");
    for (int i = 0; i < iteration; ++i) matmulNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
  }
  {
    CUProfiler profiler("shared mem");
    for (int i = 0; i < iteration; ++i) matmulShared<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
  }
  {
    CUProfiler profiler("bank opt");
    for (int i = 0; i < iteration; ++i) matmulBankOpt<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
  }

  matmulBankOpt<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
  cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  if (checkResult) {
    float* hRef = static_cast<float*>(malloc(M * N * sizeof(float)));
    matmulBase(hA, hB, hRef, M, N, K);
    matrixChecker(hRef, hC, M, N);
  }
}