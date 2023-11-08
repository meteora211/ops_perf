#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <type_traits>
#include <chrono>
#include <math.h>

class Timer {
public:
  Timer() : timer_(std::chrono::high_resolution_clock::now()) {}
  double tok() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - timer_;
    return diff.count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> timer_;
};

inline bool matrixChecker(float* res, float* expect, int M, int N) {
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

double get_matmul_FLOPs(int M, int K, int N);

double get_matmul_GFLOPS(int M, int K, int N, double time);

double get_transpose_FLOPs(int M, int K);

double get_GFLOPS(double flops, double time);

template<typename T>
void fullfill_rand(std::shared_ptr<T> input, int nelm) {
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  for (int i = 0; i < nelm; ++i) {
    input[i] = std::rand() / static_cast<std::remove_extent_t<T>>(RAND_MAX);
  }
}

template<typename T>
void fullfill_num(std::shared_ptr<T> input, int nelm, std::remove_extent_t<T> num) {
  for (int i = 0; i < nelm; ++i) {
    input[i] = num;
  }
}

template<typename T>
void print_matrix(std::shared_ptr<T> input, int nelm) {
  for (int i = 0; i < nelm; ++i) {
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;
}