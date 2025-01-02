#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>
constexpr int BLOCK_SIZE = 128;
constexpr int ELEM_PER_THREAD = 8;
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}


__device__ unsigned int coRank(const float* lhs,
                               const float* rhs,
                               int k,
                               int m,
                               int n) {
  int ilow = MAX(k - n, 0);
  int ihigh = MIN(k, m);

  while (true) {
    unsigned int i = (ilow + ihigh) / 2;
    unsigned int j = k - i;
    if ( i > 0 && j < n && lhs[i - 1] > rhs[j]) {
      ihigh = i;
    } else if (j > 0 && i < m && lhs[i] < rhs[j - 1]) {
      ilow = i;
    } else {
      return i;
    }
  }
}

__device__ void mergeSequential(const float *lhs, const float *rhs, float *res, unsigned int m,
                     unsigned int n) {
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;

  while(i < m && j < n) {
    if (lhs[i] < rhs[j]) {
      res[k++] = lhs[i++];
    } else {
      res[k++] = rhs[j++];
    }
  }
  while(i < m) {
    res[k++] = lhs[i++];
  }
  while(j < m) {
    res[k++] = rhs[j++];
  }
}

__global__ void mergeBase(const float *lhs, const float *rhs, float *res, unsigned int m,
                     unsigned int n) {
  unsigned int bSegment = blockIdx.x * blockDim.x * ELEM_PER_THREAD;
  unsigned int k = bSegment + threadIdx.x * ELEM_PER_THREAD;

  if (k < m + n) {
    unsigned int lhs_index = coRank(lhs, rhs, k, m, n);
    unsigned int rhs_index = k - lhs_index;
    float value;
    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
      if (lhs_index >= m && rhs_index >= n) break;
      if (lhs_index >= m) {
        res[k + i] = rhs[rhs_index++];
        continue;
      } else if (rhs_index >= n) {
        res[k + i] = lhs[lhs_index++];
        continue;
      }
      if (lhs[lhs_index] < rhs[rhs_index]) {
        value = lhs[lhs_index++];
      } else {
        value = rhs[rhs_index++];
      }
      res[k + i] = value;
    }
  }
}

__global__ void mergeOpt(const float *lhs, const float *rhs, float *res, unsigned int m,
                     unsigned int n) {
  __shared__ float As[ELEM_PER_THREAD * BLOCK_SIZE];
  __shared__ float Rs[ELEM_PER_THREAD * BLOCK_SIZE];

  int kBlock = blockIdx.x * blockDim.x * ELEM_PER_THREAD;
  int kBlockNext = (blockIdx.x < gridDim.x - 1) ? (kBlock + blockDim.x * ELEM_PER_THREAD) : (m + n);
  // XXX: it must be __shared__ otherwise other thread get index error.
  __shared__ int iBlock;
  __shared__ int iBlockNext;
  if (threadIdx.x == 0) {
    iBlock = coRank(lhs, rhs, kBlock, m, n);
    iBlockNext = coRank(lhs, rhs, kBlockNext, m, n);
  }
  __syncthreads();

  int lhs_elem = iBlockNext - iBlock;
  int jBlock = kBlock - iBlock;
  int jBlockNext = kBlockNext - iBlockNext;
  int rhs_elem = jBlockNext - jBlock;
  for (int i = threadIdx.x; i < lhs_elem; i += blockDim.x) {
    As[i] = lhs[iBlock + i];
  }
  float* Bs = As + lhs_elem;
  for (int i = threadIdx.x; i < rhs_elem; i += blockDim.x) {
    Bs[i] = rhs[jBlock + i];
  }
  __syncthreads();

  int k = threadIdx.x * ELEM_PER_THREAD;
  if (k < lhs_elem + rhs_elem) {
    int lhs_index = coRank(As, Bs, k, lhs_elem, rhs_elem);
    int rhs_index = k - lhs_index;
    float value;
    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
      if (lhs_index >= lhs_elem && rhs_index >= rhs_elem) break;
      if (lhs_index >= lhs_elem) {
        Rs[k + i] = Bs[rhs_index++];
        continue;
      } else if (rhs_index >= rhs_elem) {
        Rs[k + i] = As[lhs_index++];
        continue;
      }
      if (As[lhs_index] < Bs[rhs_index]) {
        value = As[lhs_index++];
      } else {
        value = Bs[rhs_index++];
      }
      Rs[k + i] = value;
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < lhs_elem + rhs_elem; i += blockDim.x) {
    res[kBlock + i] = Rs[i];
  }

}


torch::Tensor merge(const torch::Tensor &lhs, const torch::Tensor &rhs) {
  auto lhs_size = lhs.size(0);
  auto rhs_size = rhs.size(0);
  auto out_size = lhs_size + rhs_size;
  unsigned int numblock = cdiv(out_size, BLOCK_SIZE * ELEM_PER_THREAD);
  dim3 thread_per_block(BLOCK_SIZE);
  dim3 block_per_grid(numblock);
  auto res = torch::zeros({out_size}, lhs.options());
  // mergeBase<<<block_per_grid, thread_per_block>>>(
  //     lhs.data_ptr<float>(), rhs.data_ptr<float>(), res.data_ptr<float>(),
  //     lhs_size, rhs_size);
  mergeOpt<<<block_per_grid, thread_per_block>>>(
      lhs.data_ptr<float>(), rhs.data_ptr<float>(), res.data_ptr<float>(),
      lhs_size, rhs_size);

  return res;
}
