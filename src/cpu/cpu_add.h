# pragma once

// TODO: make it as a functor?
template<typename T>
void add_kernel(T* lhs_data, T* rhs_data, T* out_data,
                const int64_t M, const int64_t N) {
  // TODO: omp parallel
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      out_data[i * N + j] = lhs_data[i * N + j] + rhs_data[i * N + j];
    }
  }
}
