#include "baseline.h"
#include "cpu_matmul.h"
#include "gpu_matmul.h"
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

  auto benchmark = [&]<typename F>(F fn, int i) {
    const int nelm = i*i;
    std::shared_ptr<float[]> lhs(new float[nelm]);
    std::shared_ptr<float[]> rhs(new float[nelm]);
    std::shared_ptr<float[]> res(new float[nelm]);

    fullfill_rand(lhs, nelm);
    fullfill_rand(rhs, nelm);
    fullfill_rand(res, nelm);

    double gflops;
    if constexpr (std::is_same_v<std::invoke_result_t<F, std::shared_ptr<float[]>, std::shared_ptr<float[]>, std::shared_ptr<float[]>, int, int, int>, double>) {
      gflops = fn(lhs, rhs, res, i, i, i);
    } else {
      Timer t;
      fn(lhs, rhs, res, i, i, i);
      auto dur = t.tok();
      gflops = get_matmul_GFLOPS(i, i, i, dur);
    }
    return std::pair(i, gflops);
  };

  auto run = [&]<typename F>(F fn, const std::string& name, int size = 0){
    if (size <= 0) {
      for (const auto&& [i, gflops] : std::views::iota(0, 1000) | std::views::filter(step) | std::views::transform([&](int i){ return benchmark(fn, i); })) {
        std::cout << fmt::format("size: {}, {} gflops: {}.\n", i, name, gflops);
      }
    } else {
      std::cout << "using size: " << size << std::endl;
      // warm up
      for (int i = 0; i < 10; ++i) benchmark(fn, size);
      auto [tmp, gflops] = benchmark(fn, size);
      std::cout << fmt::format("size: {}, {} gflops: {}.\n", size, name, gflops);
    }
  };

  int size = 1000;
  // run(matmul_baseline<float[]>, "baseline", size);
  // run(matmul_transpose<float[]>, "tranpose", size);
  // run(matmul_block<float[]>, "block", size);
  // run(matmul_unroll, "unroll", size);
  // run(matmul_block_unroll, "block unroll", size);
  // run(matmul_sse, "SSE", size);
  run(matmul_cublas, "cublas", size);
  run(matmul_cuda_naive, "cuda naive", size);
  run(matmul_cuda_transpose, "cuda transpose", size);
  run(matmul_cuda_block, "cuda block", size);

  std::cout << "BENCHMARK END" << std::endl;
}
