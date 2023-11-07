#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <type_traits>
#include <chrono>

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

double get_matmul_FLOPs(int M, int K, int N);

double get_matmul_GFLOPS(int M, int K, int N, double time);

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