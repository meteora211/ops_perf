#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <vector>

#include <sys/time.h>

#include "utils.h"
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// #define BLOCK_SIZE 32
constexpr int BLOCK_SIZE = 32;
constexpr int COARSE_FACTOR = 4;

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void matmulBase(float *lhs, float *rhs, float *res, int M, int N, int K) {
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

__global__ void matmulNaive(float *A, float *B, float *C, int M, int N, int K) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
  }
}

__global__ void matmulGlobalCoalesc(float *A, float *B, float *C, int M, int N,
                                    int K) {
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

__global__ void matmulShared(float *A, float *B, float *C, int M, int N,
                             int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = threadIdx.y;
  int bj = threadIdx.x;

  float sum = 0;
  __shared__ float shareA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shareB[BLOCK_SIZE][BLOCK_SIZE];
  for (int k = 0; k < K; k += BLOCK_SIZE) {
    if ((i < M) && ((k + bj) < K)) {
      shareA[bi][bj] = A[i * K + (k + bj)];
    } else {
      shareA[bi][bj] = 0;
    }
    if ((j < N) && ((k + bi) < K)) {
      shareB[bj][bi] = B[(k + bi) * N + j]; // bank conflict
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

__global__ void matmulBankOpt(float *A, float *B, float *C, int M, int N,
                              int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int bi = threadIdx.y;
  int bj = threadIdx.x;

  float sum = 0;
  __shared__ float shareA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shareB[BLOCK_SIZE][BLOCK_SIZE + 1];
  for (int k = 0; k < K; k += BLOCK_SIZE) {
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

__global__ void matmulThreadCoarsening(float *A, float *B, float *C, int M,
                                       int N, int K) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x * COARSE_FACTOR;
  int bi = threadIdx.y;
  int bj = threadIdx.x;

  float value[COARSE_FACTOR];
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    value[c] = 0.0;
  }
  __shared__ float shareA[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ float shareB[BLOCK_SIZE][BLOCK_SIZE + 1];
  for (int k = 0; k < K; k += BLOCK_SIZE) {
    if ((i < M) && ((k + bj) < K)) {
      shareA[bi][bj] = A[i * K + (k + bj)];
    } else {
      shareA[bi][bj] = 0;
    }
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int offset = BLOCK_SIZE * c;
      if (((j + offset) < N) && ((k + bi) < K)) {
        shareB[bi][bj] = B[(k + bi) * N + (j + offset)];
      } else {
        shareB[bi][bj] = 0;
      }
      __syncthreads();
      for (int inner = 0; inner < BLOCK_SIZE; ++inner) {
        value[c] += shareA[bi][inner] * shareB[inner][bj];
      }
      __syncthreads();
    }
  }
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int col = j + BLOCK_SIZE * c;
    if (i < M && col < N) {
      C[i * N + col] = value[c];
    }
  }
}

// template <const int BM, const int BN, const int BK, const int TM>
// __global__ void matmul1DBlockTiling(float *A, float *B, float *C, int M, int
// N, int K) {
//   int i = threadIdx.y + blockIdx.y * blockDim.y;
//   int j = threadIdx.x + blockIdx.x * blockDim.x;
//   int ti = threadIdx.y;
//   int tj = threadIdx.x;
//   int bCol = blockIdx.x * blockDim.x;
//   int bRow = blockIdx.y * blockDim.y;
//
//   __shared__ float shareA[BM][BK];
//   __shared__ float shareB[BK][BN];
//
//   float threadRes[TM] = {0.0};
//   for (int k = 0; k < K; k+=BK) {
//     shareA[ti][tj] = A[i * K + (k + tj)];
//     shareB[tj][ti] = B[(k + tj) * N + (ti + bCol)];
//     __syncthreads();
//     for (int tid = 0; tid < TM; ++tid) {
//       for (int inner = 0; inner < BK; ++inner) {
//         threadRes[tid] += shareA[ti + tid][inner] * shareB[ti][inner];
//         // threadRes[tid] += 0;
//       }
//     }
//     __syncthreads();
//   }
//
//   for (int tid = 0; tid < TM; ++tid) {
//     // C[(i * TM + tid) * N + j] = threadRes[tid];
//     // C[((bRow + tj) * TM + tid) * N + (bCol + ti)] = threadRes[tid];
//     // C[(bRow + tj * TM + tid) * N + (bCol + ti)] = 0;
//     C[(bRow + tj * TM + tid) * N + (bCol + ti)] = threadRes[tid];
//   }
// }

template <const int BM, const int BN, const int BK, const int TM>
__global__ void matmul1DBlockTiling(float *A, float *B, float *C, int M, int N,
                                    int K) {
  const uint bRow = blockIdx.y;
  const uint bCol = blockIdx.x;

  __shared__ float shareA[BM * BK];
  __shared__ float shareB[BK * BN];

  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  float threadRes[TM] = {0.0};

  const uint rowA = threadIdx.x / BK;
  const uint colA = threadIdx.x % BK;
  const uint rowB = threadIdx.x / BN;
  const uint colB = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;
  const int threadCol = threadIdx.x % BN;

  for (int k = 0; k < K; k += BK) {
    shareA[rowA * BK + colA] = A[rowA * K + (colA + k)];
    shareB[rowB * BN + colB] = B[(rowB + k) * N + colB];
    __syncthreads();
    for (int tid = 0; tid < TM; ++tid) {
      for (int inner = 0; inner < BK; ++inner) {
        threadRes[tid] += shareA[(threadRow * TM + tid) * BK + inner] *
                          shareB[inner * BN + threadCol];
      }
    }
    __syncthreads();
  }

  for (int tid = 0; tid < TM; ++tid) {
    C[(threadRow * TM + tid) * N + threadCol] = threadRes[tid];
  }
}

int main() {
  const bool checkResult = false;
  const int iteration = 50;
  std::vector<int> SIZES = {128, 256, 512, 1024, 2048, 4096};
  // std::vector<int> SIZES = {4096};

  CudaDeviceInfo();

  long M, N, K;
  for (const auto &size : SIZES) {
    M = N = K = size;
    long flops = 2 * M * N * K;
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    float *hA = static_cast<float *>(malloc(M * K * sizeof(float)));
    float *hB = static_cast<float *>(malloc(K * N * sizeof(float)));
    float *hC = static_cast<float *>(malloc(M * N * sizeof(float)));
    fullfill_rand(hA, M * K);
    fullfill_rand(hB, K * N);

    float *dA;
    cudaMalloc(&dA, M * K * sizeof(float));
    float *dB;
    cudaMalloc(&dB, K * N * sizeof(float));
    float *dC;
    cudaMalloc(&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // {
    //   CUProfiler profiler("naive", iteration, flops);
    //   for (int i = 0; i < iteration; ++i)
    //     matmulNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    // }
    {
      CUProfiler profiler("global coalescing", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulGlobalCoalesc<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N,
                                                              K);
    }
    {
      CUProfiler profiler("shared mem", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulShared<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    {
      CUProfiler profiler("bank opt", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulBankOpt<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    {
      CUProfiler profiler("thread coarsening", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulThreadCoarsening<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M,
                                                                 N, K);
    }
    {

      float elapsed_time;
      cudaEvent_t beg, end;
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      const uint BM = 64;
      const uint BN = 64;
      const uint BK = 8;
      const uint TM = 8;
      dim3 threadPerBlock(BN * BM / TM);
      // dim3 blockPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
      dim3 blockPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmul1DBlockTiling<BM, BN, BK, TM>
          <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      CUProfiler profiler("1d tiling", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmul1DBlockTiling<BM, BN, BK, TM>
            <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (checkResult) {
      float *hRef = static_cast<float *>(malloc(M * N * sizeof(float)));
      matmulBase(hA, hB, hRef, M, N, K);
      matrixChecker(hRef, hC, M, N);
    }
  }
}
