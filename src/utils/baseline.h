#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <type_traits>

template<typename T>
void matmul_baseline(T* lhs, T* rhs, T* res, int M, int N, int K) {
  // lhs(M*K) * rhs(K*N) = res(M*N)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::remove_extent_t<T> sum = 0;
      for (int k = 0; k < K; ++k) {
        // sum += lhs[i, k] * rhs[k, j];
        sum += lhs[i * K + k] * rhs[k * N + j];
      }
      res[i * N + j] = sum;
    }
  }
}

template<typename T>
void transpose_baseline(T* lhs, T* res, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      res[i * M + j] = lhs[j * N + i];
    }
  }
}

template<typename T>
void reduce_baseline(T* lhs, T* res, int N) {
  res[0] = 0;
  for (int i = 0; i < N; ++i) {
    res[0] += lhs[i];
  }
}
