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

// __global__ void reduceNaive(float* lhs, float* res, int M) {
//   int i = threadIdx.x + blockDim.x * blockIdx.x;
// }


int main() {
  std::cout << "test matmul naive" << std::endl;
  int M = 3000;
  const bool checkResult = true;
  const int iteration = 50;
  // dim3 threadPerBlock(BLOCK_SIZE);
  // dim3 blockPerGrid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // 
  // float* hA = static_cast<float*>(malloc(M * sizeof(float)));
  // float* hB = static_cast<float*>(malloc(blockPerGrid.x * sizeof(float)));
  // 
  // float* dA;
  // cudaMalloc(&dA, M * sizeof(float));
  // float* dB;
  // cudaMalloc(&dB, blockPerGrid.x * sizeof(float));
  // 
  // cudaMemcpy(hA, dA, M * sizeof(float), cudaMemcpyHostToDevice);
  // 
  // reduceNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  // // warm up
  // // for (int i = 0; i < 20; ++i) {
  // //   reduceNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  // // }
  // // {
  // //   CUProfiler profiler("naive");
  // //   for (int i = 0; i < iteration; ++i) reduceNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, M);
  // // }

  // cudaMemcpy(hB, dB, blockPerGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
  // if (checkResult) {
  //   float* hCuda = static_cast<float*>(malloc(sizeof(float)));
  //   reduceBase(hB, hCuda, BLOCK_SIZE);

  //   float* hRef = static_cast<float*>(malloc(sizeof(float)));
  //   reduceBase(hA, hRef, M);

  //   matrixChecker(hCuda, hRef, 1, 1);
  // }
}