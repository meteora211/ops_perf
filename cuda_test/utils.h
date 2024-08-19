#pragma once
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <chrono>
#include <math.h>
#include <string>
#include <cuda_runtime.h>


void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

struct CUProfiler {
  CUProfiler(std::string name, long repeats = 1, long flops = 0) : name(name), repeats(repeats), flops(flops) {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
  }

  ~CUProfiler() {
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&msec, start, end);
    cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    cudaCheck(cudaGetLastError(), __FILE__, __LINE__); // Check for async errors during kernel run
    std::cout << "[" << name << "]: " << "Avg elasped time: " << time_s() << " s.";
    if (flops > 0) {
      std::cout << " FLOPS: " << GFLOPS(flops) << "." << std::endl;
    } else {
      std::cout << std::endl;
    }
  }
  float time_ms() {
    return msec / repeats;
  }
  float time_s() {
    return msec / 1000 / repeats;
  }
  float GFLOPS(long flops) {
    return flops * 1e-9 / time_s();
  }

  std::string name;
  long repeats = 1;
  long flops = 0;
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
          std::cout << "ERROR in: " << i << " , " << j << " Value: " << res[index] << " " << expect[index] << std::endl;
          correct = false;
      }
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  return correct;
}
