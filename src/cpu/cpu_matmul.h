#pragma once
#include "utils.h"
#include "baseline.h"
#include <new>
#include <emmintrin.h>
#include <immintrin.h>
#include <fmaintrin.h>

template<typename T>
void transpose(const T* src, T* dst, int row, int col);

template<typename T>
void matmul_transpose(const T* lhs, const T* rhs, T* res, int M, int N, int K);

template<typename T>
void matmul_block(const T* lhs, const T* rhs, T* res, int M, int N, int K);

void matmul_unroll(const float* lhs, const float* rhs, float* res, int M, int N, int K);

void matmul_block_unroll(const float* lhs, const float* rhs, float* res, int M, int N, int K);

void matmul_sse(const float* lhs, const float* rhs, float* res, int M, int N, int K);

// TODO: unfortunately my machine doesn't support avx512
/* void matmul_avx512(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> rhs, std::shared_ptr<float[]> res, int M, int N, int K); */

// XXX: template instantiation
// for better transpose algorithm, see:
// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
template<typename T>
void transpose(const T* src, T* dst, int row, int col) {
  for (int i = 0; i < row * col; ++i) {
    int row_idx = i / row;
    int col_idx = i % row;
    // dst[i] --> dst[row_idx, col_idx] --> src[col_idx, row_idx]
    dst[i] = src[col_idx * col + row_idx];
  }
}

template<typename T>
void matmul_transpose(const T* lhs, const T* rhs, T* res, int M, int N, int K) {
  // lhs(M*K) * rhs(K*N) = res(M*N)
  auto trans_rhs = std::shared_ptr<T>(new std::remove_extent_t<T>[N*K]);
  transpose(rhs, trans_rhs.get(), K, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::remove_extent_t<T> sum = 0;
      for (int k = 0; k < K; ++k) {
        // sum += lhs[i, k] * rhs[k, j];
        sum += lhs[i * K + k] * trans_rhs.get()[j * K + k];
      }
      res[i * N + j] = sum;
    }
  }
}

template<typename T>
void matmul_block(const T* lhs, const T* rhs, T* res, int M, int N, int K) {
  // lhs(M*K) * rhs(K*N) = res(M*N)
#ifdef __cpp_lib_hardware_interference_size
  // block_size = 64 (a cache line size) / 4 (sizeof(float)) / 2 (block for rhs and lhs) = 8
  constexpr size_t block_size = std::hardware_constructive_interference_size / sizeof(std::remove_extent_t<T>) / 2;
#else
  constexpr size_t block_size = 8;
#endif
  auto trans_rhs = std::shared_ptr<T>(new std::remove_extent_t<T>[N*K]);
  transpose(rhs, trans_rhs.get(), K, N);
  // do not clear res for pure speed test
  // fullfill_num(res, M*N, 0);

  for (int i = 0; i < M; i += block_size) {
    for (int j = 0; j < N; j += block_size) {
      for (int k = 0; k < K; k += block_size) {
        // res_b = lhs_b[i * K + k] * rhs_b[k * N + j];
        for (int bi = 0; bi < block_size; ++bi) {
          for (int bj = 0; bj < block_size; ++bj) {
            std::remove_extent_t<T> sum = 0;
            auto i_idx = i + bi;
            auto j_idx = j + bj;
            for (int bk = 0; bk < block_size; ++bk) {
              auto k_idx = k + bk;
              if (i_idx < M && j_idx < N && k_idx < K) {
                // auto rhs_idx = k_idx * N + j_idx;
                // sum += lhs[lhs_idx] * rhs[rhs_idx];
                auto lhs_idx = i_idx * K + k_idx;
                auto rhs_idx = j_idx * K + k_idx;
                sum += lhs[lhs_idx] * trans_rhs.get()[rhs_idx];
              }
            }
            // res_b[bi * block_size + bj] = sum;
            if (i_idx < M && j_idx < N) {
              res[i_idx * N + j_idx] += sum;
            }
          }
        }
      }
    }
  }
}
