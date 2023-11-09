#include "utils.h"

bool matrixChecker(float* res, float* expect, int M, int N) {
  bool correct = true;
  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6; // machine zero

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int index = i * N + j;
      double abs_err = fabs(res[index] - expect[index]);
      double dot_length = M * N;
      double abs_val = fabs(expect[index]);
      double rel_err = abs_err / abs_val / dot_length;

      if (rel_err > eps) {
          correct = false;
      }
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  return correct;
}

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