#include "utils.h"

double get_matmul_FLOPs(int M, int K, int N) {
  return static_cast<double>((K * 2.) * M * N);
}

double get_transpose_FLOPs(int M, int K) {
  return 0;
}

double get_matmul_GFLOPS(int M, int K, int N, double time) {
  return get_matmul_FLOPs(M, K, N) / 1024 / 1024 / 1024 / time;
}

double get_GFLOPS(double flops, double time) {
  return flops / 1024 / 1024 / 1024 / time;
}