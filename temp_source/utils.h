#pragma once
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <chrono>
#include <math.h>
#include <string>

struct CUProfiler {
  CUProfiler(std::string name) : name(name) {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
  }

  ~CUProfiler() {
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&msec, start, end);
    std::cout << "[" << name << "]: " << "Average elasped time: " << msec << " ms." << std::endl;
  }
  std::string name;
  float msec;
  cudaEvent_t start;
  cudaEvent_t end;
};

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
  double eps = 1.e-5; // machine zero

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int index = i * N + j;
      double abs_err = fabs(res[index] - expect[index]);
      double dot_length = M * N;
      double abs_val = fabs(expect[index]);
      double rel_err = abs_err / abs_val / dot_length;

      if (rel_err > eps) {
          std::cout << "ERROR in: " << i << " , " << j << std::endl;
          correct = false;
      }
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  return correct;
}
