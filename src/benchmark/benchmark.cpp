#include "baseline.h"
#include "cpu_matmul.h"
#include "gpu_matmul.h"
#include "gpu_transpose.h"
#include "utils.h"
#include <ranges>
#include <functional>
#include <string>
#include <fmt/core.h>
#include <type_traits>

typedef std::function<void(std::shared_ptr<float[]>, std::shared_ptr<float[]>, std::shared_ptr<float[]>, int, int, int)> Fn;
typedef std::function<double(std::shared_ptr<float[]>, std::shared_ptr<float[]>, std::shared_ptr<float[]>, int, int, int)> CuFn;

int main() {
  std::cout << "BENCHMARK START" << std::endl;

  // Use 16 to suit 4/8 alignment.
  auto step = [](int i){return i % 32 == 0 && i > 0;};
  bool checker = true;

  // TODO: combine togather
  auto benchmark_matmul = [&]<typename F>(F fn, int i) {
    const int nelm = i*i;
    std::shared_ptr<float[]> lhs(new float[nelm]);
    std::shared_ptr<float[]> rhs(new float[nelm]);
    std::shared_ptr<float[]> res(new float[nelm]);

    fullfill_rand(lhs, nelm);
    fullfill_rand(rhs, nelm);
    fullfill_rand(res, nelm);

    double dur;
    if constexpr (std::is_same_v<std::invoke_result_t<F, std::shared_ptr<float[]>, std::shared_ptr<float[]>, std::shared_ptr<float[]>, int, int, int>, double>) {
      dur = fn(lhs, rhs, res, i, i, i);
    } else {
      Timer t;
      fn(lhs, rhs, res, i, i, i);
      auto dur = t.tok();
      // gflops = get_matmul_GFLOPS(i, i, i, dur);
    }
    return std::pair(i, dur);
  };

  auto benchmark_transpose = [&]<typename F>(F fn, int i) {
    const int nelm = i*i;
    std::shared_ptr<float[]> lhs(new float[nelm]);
    std::shared_ptr<float[]> res(new float[nelm]);

    fullfill_rand(lhs, nelm);
    fullfill_rand(res, nelm);

    double dur;
    if constexpr (std::is_same_v<std::invoke_result_t<F, std::shared_ptr<float[]>, std::shared_ptr<float[]>, int, int>, double>) {
      dur = fn(lhs, res, i, i);
    } else {
      Timer t;
      fn(lhs, res, i, i);
      auto dur = t.tok();
    }
    if (checker) {
      std::shared_ptr<float[]> expect(new float[nelm]);
      transpose_baseline(lhs, expect, i, i);
      matrixChecker(res.get(), expect.get(), i, i);
    }
    return std::pair(i, dur);
  };

  auto run = [&]<typename F, typename B>(F fn, B benchmark, const std::string& name, int size = 0){
    if (size <= 0) {
      for (const auto&& [i, dur] : std::views::iota(0, 1000) | std::views::filter(step) | std::views::transform([&](int i){ return benchmark(fn, i); })) {
        std::cout << fmt::format("size: {}, {} latency: {}.\n", i, name, dur);
      }
    } else {
      std::cout << "using size: " << size << std::endl;
      // warm up
      for (int i = 0; i < 10; ++i) benchmark(fn, size);
      auto [tmp, dur] = benchmark(fn, size);
      std::cout << fmt::format("size: {}, {} latency: {}.\n", size, name, dur);
    }
  };

  int size = 1000;
  // run(matmul_baseline<float[]>, "baseline", size);
  // run(matmul_transpose<float[]>, "tranpose", size);
  // run(matmul_block<float[]>, "block", size);
  // run(matmul_unroll, "unroll", size);
  // run(matmul_block_unroll, "block unroll", size);
  // run(matmul_sse, "SSE", size);
  run(matmul_cublas, benchmark_matmul, "matmul cublas", size);
  run(matmul_cuda_naive, benchmark_matmul, "matmul cuda naive", size);
  run(matmul_cuda_transpose, benchmark_matmul, "matmul cuda transpose", size);
  run(matmul_cuda_block, benchmark_matmul, "matmul cuda block", size);

  run(transpose_cuda_naive, benchmark_transpose, "cuda naive", size);
  run(transpose_cuda_block, benchmark_transpose, "cuda block", size);
  run(transpose_cuda_bankconflict, benchmark_transpose, "cuda bankconflict", size);

  std::cout << "BENCHMARK END" << std::endl;
}
