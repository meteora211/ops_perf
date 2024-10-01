#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "utils.h"
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

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

void matmulCublas(const float *lhs, const float *rhs, float *res, int M, int N,
                  int K) {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  float cublas_alpha = 1.0;
  float cublas_beta = 0;

  // https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasSgemm#cublas-t-gemm
  cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
              // TODO: not sure why M/N/K got wrong result
              // M, N, K,
              N, M, K, &cublas_alpha, rhs, N, lhs, K, &cublas_beta, res, N);

  cublasDestroy(cublas_handle); // destroy CUBLAS context
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
        // NOTE: it's (threadRow * TM + tid) not rowA
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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul2DBlockTiling(float *A, float *B, float *C, int M, int N,
                                    int K) {
  const int bRow = blockIdx.y;
  const int bCol = blockIdx.x;

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);
  // const int shareARow = threadIdx.x / (BN / (TM * TN));
  // const int shareACol = threadIdx.x % (BN / (TM * TN));
  const int shareARow = threadIdx.x / BK;
  const int shareACol = threadIdx.x % BK;
  const int shareBRow = threadIdx.x / BN;
  const int shareBCol = threadIdx.x % BN;

  // NOTE: use blockDim.x will have performance drawback
  // const int rowStrideShareA = blockDim.x / BK;
  // const int rowStrideShareB = blockDim.x / BN;
  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
  const uint rowStrideShareA = numThreadsBlocktile / BK;
  const uint rowStrideShareB = numThreadsBlocktile / BN;

  float threadRes[TM * TN] = {0.0};
  float regA[TM] = {0.0};
  float regB[TN] = {0.0};

  __shared__ float shareA[BM * BK];
  __shared__ float shareB[BK * BN];

  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  for (int k = 0; k < K; k += BK) {
    for (int row_stride_idx = 0; row_stride_idx < BM;
         row_stride_idx += rowStrideShareA) {
      shareA[(shareARow + row_stride_idx) * BK + shareACol] =
          A[(shareARow + row_stride_idx) * K + (shareACol + k)];
    }
    for (int row_stride_idx = 0; row_stride_idx < BK;
         row_stride_idx += rowStrideShareB) {
      shareB[(shareBRow + row_stride_idx) * BN + shareBCol] =
          B[(shareBRow + k + row_stride_idx) * N + shareBCol];
    }

    __syncthreads();

    for (int inner = 0; inner < BK; ++inner) {
      for (int t_row = 0; t_row < TM; ++t_row) {
        regA[t_row] = shareA[(threadRow * TM + t_row) * BK + inner];
      }
      for (int t_col = 0; t_col < TN; ++t_col) {
        regB[t_col] = shareB[inner * BN + (threadCol * TN + t_col)];
      }
      for (int t_row = 0; t_row < TM; ++t_row) {
        for (int t_col = 0; t_col < TN; ++t_col) {
          // threadRes[t_row * TN + t_col] +=
          //     shareA[(threadRow * TM + t_row) * BK + inner] *
          //     shareB[inner * BN + (threadCol * TN + t_col)];
          threadRes[t_row * TN + t_col] += regA[t_row] * regB[t_col];
        }
      }
    }

    __syncthreads();
  }

  for (int t_row = 0; t_row < TM; ++t_row) {
    for (int t_col = 0; t_col < TN; ++t_col) {
      C[(threadRow * TM + t_row) * N + threadCol * TN + t_col] =
          threadRes[t_row * TN + t_col];
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmulVectorize(float *A, float *B, float *C, int M, int N,
                                int K) {
  const int bRow = blockIdx.y;
  const int bCol = blockIdx.x;

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);
  // const int shareARow = threadIdx.x / (BN / (TM * TN));
  // const int shareACol = threadIdx.x % (BN / (TM * TN));
  const int shareARow = threadIdx.x / (BK / 4);
  const int shareACol = threadIdx.x % (BK / 4);
  const int shareBRow = threadIdx.x / (BN / 4);
  const int shareBCol = threadIdx.x % (BN / 4);

  float threadRes[TM * TN] = {0.0};
  float regA[TM] = {0.0};
  float regB[TN] = {0.0};

  __shared__ float shareA[BM * BK];
  __shared__ float shareB[BK * BN];

  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  const uint totalResultsBlocktile = BM * BN;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
  const uint rowStrideShareA = numThreadsBlocktile / (BK / 4);
  const uint rowStrideShareB = numThreadsBlocktile / (BN / 4);

  for (int k = 0; k < K; k += BK) {
    for (int row_stride_idx = 0; row_stride_idx < BM;
         row_stride_idx += rowStrideShareA) {
      float4 tmp =
          FETCH_FLOAT4(A[(shareARow + row_stride_idx) * K + shareACol * 4]);
      shareA[(shareACol * 4 + 0) * BM + (shareARow + row_stride_idx)] = tmp.x;
      shareA[(shareACol * 4 + 1) * BM + (shareARow + row_stride_idx)] = tmp.y;
      shareA[(shareACol * 4 + 2) * BM + (shareARow + row_stride_idx)] = tmp.z;
      shareA[(shareACol * 4 + 3) * BM + (shareARow + row_stride_idx)] = tmp.w;
    }
    for (int row_stride_idx = 0; row_stride_idx < BK;
         row_stride_idx += rowStrideShareB) {
      FETCH_FLOAT4(shareB[(shareBRow + row_stride_idx) * BN + shareBCol * 4]) =
          FETCH_FLOAT4(B[(shareBRow + row_stride_idx) * N + shareBCol * 4]);
    }

    A += BK;
    B += BK * N;

    __syncthreads();

    for (int inner = 0; inner < BK; ++inner) {
      for (int t_row = 0; t_row < TM; ++t_row) {
        // regA[t_row] = shareA[inner * BM + (threadRow * TM + t_row)];
        FETCH_FLOAT4(regA[t_row]) =
            FETCH_FLOAT4(shareA[inner * BM + (threadRow * TM + t_row)]);
      }
      for (int t_col = 0; t_col < TN; ++t_col) {
        // regB[t_col] = shareB[inner * BN + (threadCol * TN + t_col)];
        FETCH_FLOAT4(regB[t_col]) =
            FETCH_FLOAT4(shareB[inner * BN + (threadCol * TN + t_col)]);
      }
      for (int t_row = 0; t_row < TM; ++t_row) {
        for (int t_col = 0; t_col < TN; ++t_col) {
          threadRes[t_row * TN + t_col] += regA[t_row] * regB[t_col];
        }
      }
    }

    __syncthreads();
  }

  for (int t_row = 0; t_row < TM; ++t_row) {
    // for (int t_col = 0; t_col < TN; ++t_col) {
    //   C[(threadRow * TM + t_row) * N + threadCol * TN + t_col] =
    //       threadRes[t_row * TN + t_col];
    // }
    for (int t_col = 0; t_col < TN; t_col += 4) {
      float4 tmp = FETCH_FLOAT4(
          C[(threadRow * TM + t_row) * N + threadCol * TN + t_col]);
      tmp.x = threadRes[t_row * TN + t_col + 0];
      tmp.y = threadRes[t_row * TN + t_col + 1];
      tmp.z = threadRes[t_row * TN + t_col + 2];
      tmp.w = threadRes[t_row * TN + t_col + 3];
      // XXX: needs to set back
      FETCH_FLOAT4(C[(threadRow * TM + t_row) * N + threadCol * TN + t_col]) =
          tmp;
    }
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int WMITER, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    matmulWrapTile(float *A, float *B, float *C, int M, int N, int K) {
  const int bRow = blockIdx.y;
  const int bCol = blockIdx.x;

  constexpr int WNITER = (BM * BN) / (TM * TN * WMITER * NUM_THREADS);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;
  constexpr int WRAPSIZE = 32;

  const int wrapIdx = threadIdx.x / WRAPSIZE;
  const int wrapRow = wrapIdx / (BN / WN);
  const int wrapCol = wrapIdx % (BN / WN);

  const int threadIdxPerWrap = threadIdx.x % WRAPSIZE;
  const int threadRow = threadIdxPerWrap / (WSUBN / TN);
  const int threadCol = threadIdxPerWrap % (WSUBN / TN);

  const int shareARow = threadIdx.x / (BK / 4);
  const int shareACol = threadIdx.x % (BK / 4);

  const int shareBRow = threadIdx.x / (BN / 4);
  const int shareBCol = threadIdx.x % (BN / 4);

  float threadRes[TM * TN * WMITER * WNITER] = {0.0};
  float regA[TM * WMITER] = {0.0};
  float regB[TN * WNITER] = {0.0};

  __shared__ float shareA[BM * BK];
  __shared__ float shareB[BK * BN];

  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  constexpr uint rowStrideShareA = NUM_THREADS / (BK / 4);
  constexpr uint rowStrideShareB = NUM_THREADS / (BN / 4);

  for (int k = 0; k < K; k += BK) {
    for (int row_stride_idx = 0; row_stride_idx < BM;
         row_stride_idx += rowStrideShareA) {
      float4 tmp =
          FETCH_FLOAT4(A[(shareARow + row_stride_idx) * K + shareACol * 4]);
      shareA[(shareACol * 4 + 0) * BM + (shareARow + row_stride_idx)] = tmp.x;
      shareA[(shareACol * 4 + 1) * BM + (shareARow + row_stride_idx)] = tmp.y;
      shareA[(shareACol * 4 + 2) * BM + (shareARow + row_stride_idx)] = tmp.z;
      shareA[(shareACol * 4 + 3) * BM + (shareARow + row_stride_idx)] = tmp.w;
    }
    for (int row_stride_idx = 0; row_stride_idx < BK;
         row_stride_idx += rowStrideShareB) {
      FETCH_FLOAT4(shareB[(shareBRow + row_stride_idx) * BN + shareBCol * 4]) =
          FETCH_FLOAT4(B[(shareBRow + row_stride_idx) * N + shareBCol * 4]);
    }

    A += BK;
    B += BK * N;

    __syncthreads();

    for (int inner = 0; inner < BK; ++inner) {
      for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
        for (int t_row = 0; t_row < TM; t_row += 4) {
          // XXX: TM is N times of 4
          FETCH_FLOAT4(regA[wm_iter * TM + t_row]) =
              FETCH_FLOAT4(shareA[inner * BM + (wrapRow * WM + wm_iter * WSUBM +
                                                threadRow * TM + t_row)]);
        }
      }
      for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
        for (int t_col = 0; t_col < TN; t_col += 4) {
          // XXX: TN is N times of 4
          FETCH_FLOAT4(regB[wn_iter * TN + t_col]) =
              FETCH_FLOAT4(shareB[inner * BN + (wrapCol * WN + wn_iter * WSUBN +
                                                threadCol * TN + t_col)]);
        }
      }
      for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
        for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
          for (int t_row = 0; t_row < TM; ++t_row) {
            for (int t_col = 0; t_col < TN; ++t_col) {
              threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                        (wn_iter * TN + t_col)] +=
                  regA[wm_iter * TM + t_row] * regB[wn_iter * TN + t_col];
            }
          }
        }
      }
    }

    __syncthreads();
  }

  for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
    for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
      for (int t_row = 0; t_row < TM; ++t_row) {

        // for (int t_col = 0; t_col < TN; ++t_col) {
        //   C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
        //     wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col] =
        //       threadRes[(wm_iter * TM + t_row) * (WNITER * TN) + (wn_iter *
        //       TN + t_col)];
        // }

        for (int t_col = 0; t_col < TN; t_col += 4) {
          float4 tmp = FETCH_FLOAT4(
              C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
                wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col]);
          tmp.x = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 0)];
          tmp.y = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 1)];
          tmp.z = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 2)];
          tmp.w = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 3)];
          FETCH_FLOAT4(
              C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
                wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col]) = tmp;
        }
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int rowStrideShareA,
          const int rowStrideShareB>
__device__ void processMemory(float *A, float *B, float *shareA, float *shareB,
                              const int shareARow, const int shareACol,
                              const int shareBRow, const int shareBCol,
                              const int N, const int K) {
  for (int row_stride_idx = 0; row_stride_idx < BM;
       row_stride_idx += rowStrideShareA) {
    float4 tmp =
        FETCH_FLOAT4(A[(shareARow + row_stride_idx) * K + shareACol * 4]);
    shareA[(shareACol * 4 + 0) * BM + (shareARow + row_stride_idx)] = tmp.x;
    shareA[(shareACol * 4 + 1) * BM + (shareARow + row_stride_idx)] = tmp.y;
    shareA[(shareACol * 4 + 2) * BM + (shareARow + row_stride_idx)] = tmp.z;
    shareA[(shareACol * 4 + 3) * BM + (shareARow + row_stride_idx)] = tmp.w;
  }
  for (int row_stride_idx = 0; row_stride_idx < BK;
       row_stride_idx += rowStrideShareB) {
    FETCH_FLOAT4(shareB[(shareBRow + row_stride_idx) * BN + shareBCol * 4]) =
        FETCH_FLOAT4(B[(shareBRow + row_stride_idx) * N + shareBCol * 4]);
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int WMITER, const int WNITER, const int WSUBM, const int WSUBN>
__device__ void processCompute(
  float* regA,
  float* regB,
  float* shareA,
  float* shareB,
  float* threadRes,
  const int wrapRow,
  const int wrapCol,
  const int threadRow,
  const int threadCol
) {
    for (int inner = 0; inner < BK; ++inner) {
      for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
        for (int t_row = 0; t_row < TM; t_row += 4) {
          // XXX: TM is N times of 4
          FETCH_FLOAT4(regA[wm_iter * TM + t_row]) =
              FETCH_FLOAT4(shareA[inner * BM + (wrapRow * WM + wm_iter * WSUBM +
                                                threadRow * TM + t_row)]);
        }
      }
      for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
        for (int t_col = 0; t_col < TN; t_col += 4) {
          // XXX: TN is N times of 4
          FETCH_FLOAT4(regB[wn_iter * TN + t_col]) =
              FETCH_FLOAT4(shareB[inner * BN + (wrapCol * WN + wn_iter * WSUBN +
                                                threadCol * TN + t_col)]);
        }
      }
      for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
        for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
          for (int t_row = 0; t_row < TM; ++t_row) {
            for (int t_col = 0; t_col < TN; ++t_col) {
              threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                        (wn_iter * TN + t_col)] +=
                  regA[wm_iter * TM + t_row] * regB[wn_iter * TN + t_col];
            }
          }
        }
      }
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int WMITER, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    matmulWrapTileDoubleBuffer(float *A, float *B, float *C, int M, int N,
                               int K) {
  const int bRow = blockIdx.y;
  const int bCol = blockIdx.x;

  constexpr int WNITER = (BM * BN) / (TM * TN * WMITER * NUM_THREADS);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;
  constexpr int WRAPSIZE = 32;

  const int wrapIdx = threadIdx.x / WRAPSIZE;
  const int wrapRow = wrapIdx / (BN / WN);
  const int wrapCol = wrapIdx % (BN / WN);

  const int threadIdxPerWrap = threadIdx.x % WRAPSIZE;
  const int threadRow = threadIdxPerWrap / (WSUBN / TN);
  const int threadCol = threadIdxPerWrap % (WSUBN / TN);

  const int shareARow = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const int shareACol = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);

  const int shareBRow = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const int shareBCol = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);

  float threadRes[TM * TN * WMITER * WNITER] = {0.0};
  float regA[TM * WMITER] = {0.0};
  float regB[TN * WNITER] = {0.0};

  __shared__ float shareA[2 * BM * BK];
  __shared__ float shareB[2 * BK * BN];

  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  const uint totalResultsBlocktile = BM * BN;
  constexpr uint rowStrideShareA = (NUM_THREADS / 2) / (BK / 4);
  constexpr uint rowStrideShareB = (NUM_THREADS / 2) / (BN / 4);
  int bufferIdx = threadIdx.x / (NUM_THREADS / 2);

  if (bufferIdx == 0) {
    processMemory<BM, BN, BK, rowStrideShareA, rowStrideShareB>(
        A, B, shareA, shareB,
        shareARow, shareACol, shareBRow, shareBCol, N, K);
  }
  __syncthreads();
  for (int k = 0; k < K; k += 2 * BK) {
    if (bufferIdx == 0) {
      processCompute<BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER, WSUBM, WSUBN>(
        regA, regB, shareA, shareB, threadRes, wrapRow, wrapCol, threadRow, threadCol
      );
      __syncthreads();

      if (k + BK < K) {
        processCompute<BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER, WSUBM, WSUBN>(
          regA, regB, shareA + BM * BK, shareB + BK * BN, threadRes, wrapRow, wrapCol, threadRow, threadCol
        );
      }
      __syncthreads();

      if (k + 2*BK < K) {
        processMemory<BM, BN, BK, rowStrideShareA, rowStrideShareB>(
            A + 2 * BK, B + 2 * BK * N, shareA, shareB,
            shareARow, shareACol, shareBRow, shareBCol, N, K);

      }

    } else {
      if (k + BK < K) {
        processMemory<BM, BN, BK, rowStrideShareA, rowStrideShareB>(
            A + BK, B + BK * N, shareA + BM * BK, shareB + BK * BN,
            shareARow, shareACol, shareBRow, shareBCol, N, K);
      }
      __syncthreads();

      processCompute<BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER, WSUBM, WSUBN>(
        regA, regB, shareA, shareB, threadRes, wrapRow, wrapCol, threadRow, threadCol
      );
      __syncthreads();

      if (k + BK < K) {
        processCompute<BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER, WSUBM, WSUBN>(
          regA, regB, shareA + BM * BK, shareB + BK * BN, threadRes, wrapRow, wrapCol, threadRow, threadCol
        );
      }
    }
    __syncthreads();
    A += 2 * BK;
    B += 2 * BK * N;
  }

  for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
    for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
      for (int t_row = 0; t_row < TM; ++t_row) {
        for (int t_col = 0; t_col < TN; t_col += 4) {
          float4 tmp = FETCH_FLOAT4(
              C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
                wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col]);
          tmp.x = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 0)];
          tmp.y = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 1)];
          tmp.z = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 2)];
          tmp.w = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 3)];
          FETCH_FLOAT4(
              C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
                wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col]) = tmp;
        }
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int WMITER, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    matmulWrapTileDoubleBufferV2(float *A, float *B, float *C, int M, int N,
                                 int K) {
  const int bRow = blockIdx.y;
  const int bCol = blockIdx.x;

  constexpr int WNITER = (BM * BN) / (TM * TN * WMITER * NUM_THREADS);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;
  constexpr int WRAPSIZE = 32;

  const int wrapIdx = threadIdx.x / WRAPSIZE;
  const int wrapRow = wrapIdx / (BN / WN);
  const int wrapCol = wrapIdx % (BN / WN);

  const int threadIdxPerWrap = threadIdx.x % WRAPSIZE;
  const int threadRow = threadIdxPerWrap / (WSUBN / TN);
  const int threadCol = threadIdxPerWrap % (WSUBN / TN);

  const int shareARow = threadIdx.x / (BK / 4);
  const int shareACol = threadIdx.x % (BK / 4);

  const int shareBRow = threadIdx.x / (BN / 4);
  const int shareBCol = threadIdx.x % (BN / 4);

  float threadRes[TM * TN * WMITER * WNITER] = {0.0};
  float regA[TM * WMITER] = {0.0};
  float regB[TN * WNITER] = {0.0};

  __shared__ float shareA[2 * BM * BK];
  __shared__ float shareB[2 * BK * BN];

  A += bRow * BM * K;
  B += bCol * BN;
  C += bRow * BM * N + bCol * BN;

  const uint totalResultsBlocktile = BM * BN;
  constexpr uint rowStrideShareA = NUM_THREADS / (BK / 4);
  constexpr uint rowStrideShareB = NUM_THREADS / (BN / 4);

  processMemory<BM, BN, BK, rowStrideShareA, rowStrideShareB>(
      A, B, shareA, shareB,
      shareARow, shareACol, shareBRow, shareBCol, N, K);
  __syncthreads();
  int write_index = 1;
  for (int k = 0; k < K; k += BK) {
    int load_index = write_index ^ 1;
    processCompute<BM, BN, BK, WM, WN, TM, TN, WMITER, WNITER, WSUBM, WSUBN>(
      regA, regB, shareA + load_index * BM * BK, shareB + load_index * BK * BN,
      threadRes, wrapRow, wrapCol, threadRow, threadCol
    );

    if (k + BK < K) {
      processMemory<BM, BN, BK, rowStrideShareA, rowStrideShareB>(
          A + BK, B + BK * N, shareA + write_index * BM * BK, shareB + write_index * BK * BN,
          shareARow, shareACol, shareBRow, shareBCol, N, K);
    }

    __syncthreads();
    A += BK;
    B += BK * N;
    write_index ^= 1;
  }

  for (int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
    for (int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
      for (int t_row = 0; t_row < TM; ++t_row) {
        for (int t_col = 0; t_col < TN; t_col += 4) {
          float4 tmp = FETCH_FLOAT4(
              C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
                wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col]);
          tmp.x = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 0)];
          tmp.y = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 1)];
          tmp.z = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 2)];
          tmp.w = threadRes[(wm_iter * TM + t_row) * (WNITER * TN) +
                            (wn_iter * TN + t_col + 3)];
          FETCH_FLOAT4(
              C[(wrapRow * WM + wm_iter * WSUBM + threadRow * TM + t_row) * N +
                wrapCol * WN + wn_iter * WSUBN + threadCol * TN + t_col]) = tmp;
        }
      }
    }
  }
}


int main() {
  const bool checkResult = false;
  const int iteration = 50;
  std::vector<int> SIZES = {128, 256, 512, 1024, 2048, 4096};
  // std::vector<int> SIZES = {4096};
  // std::vector<int> SIZES = {512};

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

    {
      CUProfiler profiler("naive", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    {
      CUProfiler profiler("cublas", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulCublas(dA, dB, dC, M, N, K);
    }
    {
      CUProfiler profiler("global coalescing", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulGlobalCoalesc<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M,
        N,
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
        matmulThreadCoarsening<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC,
        M,
                                                                 N, K);
    }
    {
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
    {
      const uint BM = 128;
      const uint BN = 128;
      // const uint BM = 64;
      // const uint BN = 64;
      const uint BK = 8;
      const uint TM = 8;
      const uint TN = 8;
      dim3 threadPerBlock(BN * BM / (TM * TN));
      // dim3 blockPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
      dim3 blockPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmul2DBlockTiling<BM, BN, BK, TM, TN>
          <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      CUProfiler profiler("2d tiling", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmul2DBlockTiling<BM, BN, BK, TM, TN>
            <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    {
      // const uint BM = 64;
      // const uint BN = 64;
      const uint BM = 128;
      const uint BN = 128;
      const uint BK = 8;
      const uint TM = 8;
      const uint TN = 8;
      dim3 threadPerBlock(BN * BM / (TM * TN));
      // dim3 blockPerGrid((N + BN - 1) / BN, (M + BM - 1) / BM);
      dim3 blockPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmulVectorize<BM, BN, BK, TM, TN>
          <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      CUProfiler profiler("vectorize", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulVectorize<BM, BN, BK, TM, TN>
            <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    {
      // const uint BM = 64;
      // const uint BN = 64;
      const uint NUM_THREADS = 128;
      const uint BM = 128;
      const uint BN = 128;
      const uint BK = 16;
      const uint WM = 64;
      const uint WN = 64;
      const uint TM = 8;
      const uint TN = 4;
      const uint WMITER = 1;
      // const uint WNITER = 4;
      dim3 threadPerBlock(NUM_THREADS);
      dim3 blockPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmulWrapTile<BM, BN, BK, WM, WN, TM, TN, WMITER, NUM_THREADS>
          <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      CUProfiler profiler("wrap tile", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulWrapTile<BM, BN, BK, WM, WN, TM, TN, WMITER, NUM_THREADS>
            <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    }
    {
      // const uint BM = 64;
      // const uint BN = 64;
      const uint NUM_THREADS = 128;
      const uint BM = 128;
      const uint BN = 128;
      const uint BK = 16;
      const uint WM = 64;
      const uint WN = 64;
      const uint TM = 8;
      const uint TN = 4;
      const uint WMITER = 1;
      // const uint WNITER = 4;
      dim3 threadPerBlock(NUM_THREADS);
      dim3 blockPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      matmulWrapTileDoubleBufferV2<BM, BN, BK, WM, WN, TM, TN, WMITER, NUM_THREADS>
          <<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      CUProfiler profiler("wrap tile double buffer", iteration, flops);
      for (int i = 0; i < iteration; ++i)
        matmulWrapTileDoubleBufferV2<BM, BN, BK, WM, WN, TM, TN, WMITER, NUM_THREADS>
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
