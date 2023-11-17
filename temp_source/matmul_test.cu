#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <chrono>
#include <math.h>

#define BLOCK_SIZE 16

void fullfill_rand(float* input, int nelm) {
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  for (int i = 0; i < nelm; ++i) {
    input[i] = std::rand() / static_cast<float>(RAND_MAX);
  }
}

bool matrixChecker(float* res, float* expect, int M, int N) {
  bool correct = true;
  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6; // machine zero

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int index = i * N + j;
      double abs_err = fabs(res[index] - expect[index]);
      double dot_length = M * N;
      double abs_val = fabs(expect[index]);
      double rel_err = abs_err / abs_val / dot_length;

      if (rel_err > eps) {
          correct = false;
      }
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  return correct;
}

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
      sum += A[i * M + k] * B[k * N + j];
    }
    C[i * M + j] = sum;
  }
}

__global__ void matmulShared(float* A, float* B, float* C, int M, int N, int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = threadIdx.x;
  int bj = threadIdx.y;
  
  if (i < M && j < N) {
    float sum = 0;
    __shared__ float shareA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shareB[BLOCK_SIZE][BLOCK_SIZE];
    for (int k = 0; k < K; k+=BLOCK_SIZE) {
      shareA[bi][bj] = A[i * M + k];
      shareB[bj][bi] = B[k * N + j];
      __syncthreads();
      for (int inner = 0; inner < BLOCK_SIZE; ++inner) {
        sum += shareA[bi][inner] * shareB[bj][inner];
      }
      __syncthreads();
    }
    C[i * M + j] = sum;
  }
}

int main() {
  std::cout << "test matmul naive" << std::endl;
  int M = 1024;
  int N = 3000;
  int K = 2000;
  const bool checkResult = true;
  
  float* hA = static_cast<float*>(malloc(M * K * sizeof(float)));
  float* hB = static_cast<float*>(malloc(K * N * sizeof(float)));
  float* hC = static_cast<float*>(malloc(M * N * sizeof(float)));
  
  float* dA;
  cudaMalloc(&dA, M * K * sizeof(float));
  float* dB;
  cudaMalloc(&dB, K * N * sizeof(float));
  float* dC;
  cudaMalloc(&dC, M * N * sizeof(float));
  
  cudaMemcpy(hA, dA, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(hB, dB, K * N * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blockPerGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  matmulShared<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);

  cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  if (checkResult) {
    float* hRef = static_cast<float*>(malloc(M * N * sizeof(float)));
    matmulBase(hA, hB, hRef, M, N, K);
    matrixChecker(hRef, hC, M, N);
  }
}