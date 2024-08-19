#include <iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <vector>

#include "utils.h"
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// #define BLOCK_SIZE 32
constexpr int BLOCK_SIZE = 32;
constexpr int COARSE_FACTOR = 4;


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

__global__ void matmulGlobalCoalesc(float* A, float* B, float* C, int M, int N, int K) {
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

__global__ void matmulThreadCoarsening(float* A, float* B, float* C, int M, int N, int K) {
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
  for (int k = 0; k < K; k+=BLOCK_SIZE) {
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
// __global__ void matmul1DBlockTiling(float *A, float *B, float *C, int M, int N, int K) {
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

// template <const int BM, const int BN, const int BK, const int TM>
// __global__ void matmul1DBlockTiling(float *A, float *B, float *C, int M, int N, int K) {
//   const uint bRow = blockIdx.y;
//   const uint bCol = blockIdx.x;
//
//   __shared__ float shareA[BM * BK];
//   __shared__ float shareB[BK * BN];
//
//   A += bRow * BM * K;
//   B += bCol * BN;
//   C += bRow * BM * N + bCol * BN;
//
//   float threadRes[TM] = {0.0};
//
//   const uint rowA = threadIdx.x / BK;
//   const uint colA = threadIdx.x % BK;
//   const uint rowB = threadIdx.x / BN;
//   const uint colB = threadIdx.x % BN;
//   const int threadRow = threadIdx.x / BN;
//   const int threadCol = threadIdx.x % BN;
//
//   for (int k = 0; k < K; k+=BK) {
//     shareA[rowA * BK + colA] = A[rowA * K + (colA + k)];
//     shareB[rowB * BN + colB] = B[(rowB + k) * N + colB];
//     __syncthreads();
//     for (int tid = 0; tid < TM; ++tid) {
//       for (int inner = 0; inner < BK; ++inner) {
//         threadRes[tid] += shareA[(threadRow * TM + tid) * BK + inner] * shareB[inner * BN + threadCol];
//       }
//     }
//     __syncthreads();
//   }
//
//   for (int tid = 0; tid < TM; ++tid) {
//     C[(threadRow * TM + tid) * N + threadCol] = threadRes[tid];
//   }
// }

__global__ void mysgemm_v2(float *A, float *B, float *C, int M, int N, int K) {
  // int i = threadIdx.y + blockIdx.y * blockDim.y;
  // int j = threadIdx.x + blockIdx.x * blockDim.x;
  int bj = blockIdx.x;
  int bi = blockIdx.y;

  const int BM = BLOCK_SIZE;
  const int BN = BLOCK_SIZE;
  const int BK = BLOCK_SIZE;
  
  int tx = threadIdx.x % BN;
  int ty = threadIdx.x / BN;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  // 移动到当前block
  A = &A[bi * BM * K];
  B = &B[bj * BN];
  C = &C[bi * BM * N + bj * BN];

  float tmp = 0.;
  for (int k = 0; k < K; k += BK) {
      // 缓存A_tile和B_tile
      As[ty][tx] = A[ty * K + tx];
      Bs[ty][tx] = B[ty * N + tx];
      // 同步所有线程缓存完成
      __syncthreads();
      A += BK;
      B += BK * N;
      for (int i = 0; i < BK; i++) {
          tmp += As[ty][i] * Bs[i][tx];
      }
      // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
      __syncthreads();
  }
  C[ty * N + tx] = tmp + C[ty * N + tx];
}

__global__ void sgemm_shared_mem_block(const float *A, const float *B,float *C,
                                       int M, int N, int K) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCK_SIZE;
  const uint threadRow = threadIdx.x / BLOCK_SIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCK_SIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCK_SIZE;                        // row=0, col=cCol
  C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
             Bs[dotIdx * BLOCK_SIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      tmp + C[threadRow * N + threadCol];
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void matmul1DBlockTiling(float *A, float *B, float *C, int M, int N, int K) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        threadResults[resIdx];
  }
}


int main() {
  const bool checkResult = false;
  const int iteration = 50;
  std::vector<int> SIZES = {128, 256, 512, 1024, 2048, 4096};
  // std::vector<int> SIZES = {4096};

  CudaDeviceInfo();

  long M, N, K;
  // std::cout << M << " " << N << " " << K << " " << flops << " " << sizeof(long) << std::endl;
  // warm up
  // for (int i = 0; i < 20; ++i) {
  //   matmulNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, 1024, 1024, 1024);
  // }
  for (const auto& size : SIZES) {
    M = N = K = size;
    long flops = 2 * M * N * K;
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
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
  
    // {
    //   CUProfiler profiler("naive", iteration, flops);
    //   for (int i = 0; i < iteration; ++i) matmulNaive<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    // }
    // {
    //   CUProfiler profiler("global coalescing", iteration, flops);
    //   for (int i = 0; i < iteration; ++i) matmulGlobalCoalesc<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    // }
    // {
    //   CUProfiler profiler("shared mem", iteration, flops);
    //   for (int i = 0; i < iteration; ++i) matmulShared<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    // }
    // {
    //   CUProfiler profiler("bank opt", iteration, flops);
    //   for (int i = 0; i < iteration; ++i) matmulBankOpt<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    // }
    // {
    //   CUProfiler profiler("thread coarsening", iteration, flops);
    //   for (int i = 0; i < iteration; ++i) matmulThreadCoarsening<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    // }
    {
      CUProfiler profiler("1d tiling", iteration, flops);

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
      matmul1DBlockTiling<BM, BN, BK, TM><<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      cudaEventRecord(beg);
      for (int i = 0; i < iteration; ++i) matmul1DBlockTiling<BM, BN, BK, TM><<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / iteration,
          (iteration * flops * 1e-9) / elapsed_time, M);
    }
    // matmulBankOpt<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (checkResult) {
      float* hRef = static_cast<float*>(malloc(M * N * sizeof(float)));
      matmulBase(hA, hB, hRef, M, N, K);
      matrixChecker(hRef, hC, M, N);
    }
  }
  // {
  //   CUProfiler profiler("1111", flops * iteration);
  //   cudaFuncSetAttribute(sgemm_shared_mem_block,
  //                        cudaFuncAttributePreferredSharedMemoryCarveout,
  //                        cudaSharedmemCarveoutMaxShared);
  //   for (int i = 0; i < iteration; ++i) sgemm_shared_mem_block<<<blockPerGrid, threadPerBlock>>>(dA, dB, dC, M, N, K);
  // }

}
