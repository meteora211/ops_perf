#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils.h"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
//
// Original vector addition kernel without coarsening
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

// Vector addition kernel with thread coarsening
// Assuming a coarsening factor of 2
__global__ void VecAddCoarsened(float* A, float* B, float* C, int N)
{
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Coarsening factor applied here
  if (i < N)
    C[i] = A[i] + B[i];
  if (i + 1 < N) // Handle the additional element due to coarsening
    C[i + 1] = A[i + 1] + B[i + 1];
}

__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = in[(index * 2) % n];
  }
}

__global__ void copyDataCoalesced(float *in, float *out, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = in[index];
  }
}

__global__ void processArrayWithDivergence(int *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    if (data[idx] % 2 == 0) {
      data[idx] = data[idx] * 2;
    } else {
      data[idx] = data[idx] + 1;
    }
  }
}

__global__ void processArrayWithoutDivergence(int *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int isEven = !(data[idx] % 2);
    data[idx] = isEven * (data[idx] * 2) + (!isEven) * (data[idx] + 1);
  }
}

void initializeArray(int *arr, int n) {
  for(int i = 0; i < n; ++i) {
    arr[i] = i;
  }
}
void initializeArray(float *arr, int n) {
  for(int i = 0; i < n; ++i) {
    arr[i] = static_cast<float>(i);
  }
}
void random_init(float* data, int size) {
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}


int main() {
  float *in, *out;
  int* in_int;
  std::vector<int> SIZES = {4096};
  for (const auto &size : SIZES) {
    cudaMallocManaged(&in, size * sizeof(float));
    cudaMallocManaged(&in_int, size * sizeof(int));
    cudaMallocManaged(&out, size * sizeof(float));
    initializeArray(in, size);
    initializeArray(in_int, size);
    int blockSize = 1024; // Optimal block size for many devices
    int numBlocks = (size + blockSize - 1) / blockSize; // Calculate the number of blocks
    int minGridSize = 40;

    // Optimize grid dimensions based on device properties
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, copyDataCoalesced, 0, 0);
    // Print suggested block size and minimum grid size
    std::cout << "Recommended block size: " << blockSize
              << ", Minimum grid size: " << minGridSize << std::endl;

    numBlocks = (size + blockSize - 1) / blockSize;
    // FIXME: data is cached and re-create data for each tests
    {
      CUProfiler profiler("non-coalesced");
      // Launch non-coalesced kernel
      copyDataNonCoalesced<<<numBlocks, blockSize>>>(in, out, size);
    }
    {
      CUProfiler profiler("coalesced");
      // Launch coalesced kernel
      copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, size);
    }
    {
      CUProfiler profiler("non-divergence");
      // Launch non divergence
      processArrayWithoutDivergence<<<numBlocks, blockSize>>>(in_int, size);
    }
    {
      CUProfiler profiler("divergence");
      // Launch divergence
      processArrayWithDivergence<<<numBlocks, blockSize>>>(in_int, size);
    }
    {
      CUProfiler profiler("non-coarsened");
      VecAdd<<<numBlocks, blockSize>>>(in, in, in, size);
    }
    {
      CUProfiler profiler("coarsened");
      VecAddCoarsened<<<numBlocks, blockSize>>>(in, in, in, size);
    }

    cudaFree(in);
    cudaFree(in_int);
    cudaFree(out);

  }

  return 0;
}
