// #include "gpu_transpose.h"
#include "utils.h"
#include "baseline.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define OFFSET(a, b, c) a*c+b

double transpose_cuda_executor(void(*cuda_func)(float *, float *, const int, const int),
                   std::shared_ptr<float[]> lhs,
                   std::shared_ptr<float[]> res,
                   int M, int N,
                   dim3 grid,
                   dim3 block) {
  size_t lhs_size = M * N * sizeof(float);
  size_t res_size = M * N * sizeof(float);

  float* lhs_device, *res_device;
  cudaMalloc(&lhs_device, lhs_size);
  cudaMalloc(&res_device, res_size);

  cudaMemcpy(lhs_device, lhs.get(), lhs_size, cudaMemcpyHostToDevice);
  cudaMemcpy(res_device, res.get(), res_size, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  cuda_func<<<grid, block>>>(lhs_device, res_device, M, N);

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec;
  cudaEventElapsedTime(&msec, start, end);
  // auto gflops = get_GFLOPS(get_transpose_FLOPs(M, N), msec/1000);

  cudaMemcpy(res.get(), res_device, res_size, cudaMemcpyDeviceToHost);


  cudaFree(lhs_device);
  cudaFree(res_device);
  return msec;
}

__global__ void transpose_naive(float* lhs, float* res, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    res[OFFSET(j, i, N)] = lhs[OFFSET(i, j, M)];
    // res[OFFSET(i, j, N)] = lhs[OFFSET(j, i, M)];
}

template<int W>
__global__ void transpose_block(float* lhs, float* res, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int inIndex = OFFSET(i, j, N);
    
    // 这里简单理解就是，合并访存需要对res[i,j]做row major写，对于每一个block内的thread:
    // res[ti][tj] = tile[tj][ti]
    // 因此在thread level，res已经被transpose了，所以在block内的索引仍然是：
    // Row(i) <-> threadIdx.y
    // Col(j) <-> threadIdx.x
    // 但是grid范围下，每一个block是需要被transpose的, grid内的索引仍然是：
    // Row(i) <-> blockIdx.x * blockDim.x
    // Col(j) <-> blockIdx.y * blockDim.y
    i = blockIdx.x * blockDim.x + threadIdx.y;
    j = blockIdx.y * blockDim.y + threadIdx.x;
    int outIndex = OFFSET(i, j, M);
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    
    __shared__ float tile[W][W];
    // 访存合并，每个wrap先将数据读到shared_memory, 然后从share_memory写回去
    tile[ti][tj] =  lhs[inIndex];
    
    __syncthreads();

    // res[OFFSET(j, i, M)] = tile[OFFSET(ti, tj, W)];
    res[outIndex] = tile[tj][ti];
}

template<int W>
__global__ void transpose_bankconflict(float* lhs, float* res, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int inIndex = OFFSET(i, j, N);

    i = blockIdx.x * blockDim.x + threadIdx.y;
    j = blockIdx.y * blockDim.y + threadIdx.x;
    int outIndex = OFFSET(i, j, M);
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    
    __shared__ float tile[W][W+1];
    tile[ti][tj] =  lhs[inIndex];
    
    __syncthreads();

    // res[outIndex] = tile[OFFSET(tj, ti, W)];
    res[outIndex] = tile[tj][ti];
}

double transpose_cuda_naive(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> res, int M, int N) {
  const int BM = 32;
  const int BN = 32;

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BN, BM);
  return transpose_cuda_executor(transpose_naive, lhs, res, M, N, grid, block);
}

double transpose_cuda_block(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> res, int M, int N) {
  const int BM = 32;
  const int BN = 32;

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BN, BM);
  return transpose_cuda_executor(transpose_block<32>, lhs, res, M, N, grid, block);
}

double transpose_cuda_bankconflict(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> res, int M, int N) {
  const int BM = 32;
  const int BN = 32;

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(BN, BM);
  return transpose_cuda_executor(transpose_bankconflict<32>, lhs, res, M, N, grid, block);
}
