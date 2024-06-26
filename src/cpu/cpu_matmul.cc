#include "utils.h"
#include "cpu_matmul.h"

void matmul_unroll(const float* lhs, const float* rhs, float* res, int M, int N, int K) {
  float tmp00, tmp01, tmp02, tmp03;
  float tmp10, tmp11, tmp12, tmp13;
  float tmp20, tmp21, tmp22, tmp23;
  float tmp30, tmp31, tmp32, tmp33;

  int M_4 = (M >> 2) << 2;
  int N_4 = (N >> 2) << 2;
  for (int i = 0; i < M_4; i+=4) {
    for (int j = 0; j < N_4; j += 4) {
      tmp00 = tmp01 = tmp02 = tmp03 = 0;
      tmp10 = tmp11 = tmp12 = tmp13 = 0;
      tmp20 = tmp21 = tmp22 = tmp23 = 0;
      tmp30 = tmp31 = tmp32 = tmp33 = 0;
      for (int k = 0; k < K; ++k) {
        tmp00 += lhs[(i + 0) * K + k] * rhs[k * N + (j + 0)];
        tmp01 += lhs[(i + 0) * K + k] * rhs[k * N + (j + 1)];
        tmp02 += lhs[(i + 0) * K + k] * rhs[k * N + (j + 2)];
        tmp03 += lhs[(i + 0) * K + k] * rhs[k * N + (j + 3)];
        tmp10 += lhs[(i + 1) * K + k] * rhs[k * N + (j + 0)];
        tmp11 += lhs[(i + 1) * K + k] * rhs[k * N + (j + 1)];
        tmp12 += lhs[(i + 1) * K + k] * rhs[k * N + (j + 2)];
        tmp13 += lhs[(i + 1) * K + k] * rhs[k * N + (j + 3)];
        tmp20 += lhs[(i + 2) * K + k] * rhs[k * N + (j + 0)];
        tmp21 += lhs[(i + 2) * K + k] * rhs[k * N + (j + 1)];
        tmp22 += lhs[(i + 2) * K + k] * rhs[k * N + (j + 2)];
        tmp23 += lhs[(i + 2) * K + k] * rhs[k * N + (j + 3)];
        tmp30 += lhs[(i + 3) * K + k] * rhs[k * N + (j + 0)];
        tmp31 += lhs[(i + 3) * K + k] * rhs[k * N + (j + 1)];
        tmp32 += lhs[(i + 3) * K + k] * rhs[k * N + (j + 2)];
        tmp33 += lhs[(i + 3) * K + k] * rhs[k * N + (j + 3)];
      }
      res[(i + 0) * N + (j + 0)] = tmp00;
      res[(i + 0) * N + (j + 1)] = tmp01;
      res[(i + 0) * N + (j + 2)] = tmp02;
      res[(i + 0) * N + (j + 3)] = tmp03;

      res[(i + 1) * N + (j + 0)] = tmp10;
      res[(i + 1) * N + (j + 1)] = tmp11;
      res[(i + 1) * N + (j + 2)] = tmp12;
      res[(i + 1) * N + (j + 3)] = tmp13;

      res[(i + 2) * N + (j + 0)] = tmp20;
      res[(i + 2) * N + (j + 1)] = tmp21;
      res[(i + 2) * N + (j + 2)] = tmp22;
      res[(i + 2) * N + (j + 3)] = tmp23;

      res[(i + 3) * N + (j + 0)] = tmp30;
      res[(i + 3) * N + (j + 1)] = tmp31;
      res[(i + 3) * N + (j + 2)] = tmp32;
      res[(i + 3) * N + (j + 3)] = tmp33;
    }
  }

  if (M_4 == M && N_4 == N) {
    return;
  }

  for (int i = 0; i < M; ++i) {
    for (int j = N_4; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += lhs[i * K + k] * rhs[k * N + j];
      }
      res[i * N + j] = sum;
    }
  }
  for (int i = M_4; i < M; ++i) {
    for (int j = 0; j < N_4; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += lhs[i * K + k] * rhs[k * N + j];
      }
      res[i * N + j] = sum;
    }
  }
}

void matmul_block_unroll(const float* lhs, const float* rhs, float* res, int M, int N, int K) {
  float tmp00, tmp01, tmp02, tmp03;
  float tmp10, tmp11, tmp12, tmp13;
  float tmp20, tmp21, tmp22, tmp23;
  float tmp30, tmp31, tmp32, tmp33;
  constexpr size_t block_size = 8;
  auto trans_rhs = std::shared_ptr<float[]>(new float[N*K]);
  transpose(rhs, trans_rhs.get(), K, N);
  // do not clear res for pure speed test
  // fullfill_num(res, M*N, 0);

  // FIXME: boundary check in innerloop is skipped.
  for (int i = 0; i < M; i += block_size) {
    for (int j = 0; j < N; j += block_size) {
      for (int k = 0; k < K; k += block_size) {
        // res_b = lhs_b[i * K + k] * rhs_b[k * N + j];
        for (int bi = 0; bi < block_size; bi+=4) {
          for (int bj = 0; bj < block_size; bj+=4) {
            tmp00 = tmp01 = tmp02 = tmp03 = 0;
            tmp10 = tmp11 = tmp12 = tmp13 = 0;
            tmp20 = tmp21 = tmp22 = tmp23 = 0;
            tmp30 = tmp31 = tmp32 = tmp33 = 0;
            auto i_idx = i + bi;
            auto j_idx = j + bj;
            for (int bk = 0; bk < block_size; ++bk) {
              auto k_idx = k + bk;
              if (i_idx < M && j_idx < N && k_idx < K) {
                // auto rhs_idx = k_idx * N + j_idx;
                // sum += lhs[lhs_idx] * rhs[rhs_idx];
                auto lhs_idx_0 = (i_idx + 0) * K + k_idx;
                auto lhs_idx_1 = (i_idx + 1) * K + k_idx;
                auto lhs_idx_2 = (i_idx + 2) * K + k_idx;
                auto lhs_idx_3 = (i_idx + 3) * K + k_idx;

                auto rhs_idx_0 = (j_idx + 0) * K + k_idx;
                auto rhs_idx_1 = (j_idx + 1) * K + k_idx;
                auto rhs_idx_2 = (j_idx + 2) * K + k_idx;
                auto rhs_idx_3 = (j_idx + 3) * K + k_idx;

                /* sum += lhs[lhs_idx] * trans_rhs[rhs_idx]; */
                tmp00 += lhs[lhs_idx_0] * trans_rhs[rhs_idx_0];
                tmp01 += lhs[lhs_idx_0] * trans_rhs[rhs_idx_1];
                tmp02 += lhs[lhs_idx_0] * trans_rhs[rhs_idx_2];
                tmp03 += lhs[lhs_idx_0] * trans_rhs[rhs_idx_3];

                tmp10 += lhs[lhs_idx_1] * trans_rhs[rhs_idx_0];
                tmp11 += lhs[lhs_idx_1] * trans_rhs[rhs_idx_1];
                tmp12 += lhs[lhs_idx_1] * trans_rhs[rhs_idx_2];
                tmp13 += lhs[lhs_idx_1] * trans_rhs[rhs_idx_3];

                tmp20 += lhs[lhs_idx_2] * trans_rhs[rhs_idx_0];
                tmp21 += lhs[lhs_idx_2] * trans_rhs[rhs_idx_1];
                tmp22 += lhs[lhs_idx_2] * trans_rhs[rhs_idx_2];
                tmp23 += lhs[lhs_idx_2] * trans_rhs[rhs_idx_3];

                tmp30 += lhs[lhs_idx_3] * trans_rhs[rhs_idx_0];
                tmp31 += lhs[lhs_idx_3] * trans_rhs[rhs_idx_1];
                tmp32 += lhs[lhs_idx_3] * trans_rhs[rhs_idx_2];
                tmp33 += lhs[lhs_idx_3] * trans_rhs[rhs_idx_3];
              }
            }
            // res_b[bi * block_size + bj] = sum;
            if (i_idx < M && j_idx < N) {
              res[(i_idx + 0) * N + (j_idx + 0)] += tmp00;
              res[(i_idx + 0) * N + (j_idx + 1)] += tmp01;
              res[(i_idx + 0) * N + (j_idx + 2)] += tmp02;
              res[(i_idx + 0) * N + (j_idx + 3)] += tmp03;

              res[(i_idx + 1) * N + (j_idx + 0)] += tmp10;
              res[(i_idx + 1) * N + (j_idx + 1)] += tmp11;
              res[(i_idx + 1) * N + (j_idx + 2)] += tmp12;
              res[(i_idx + 1) * N + (j_idx + 3)] += tmp13;

              res[(i_idx + 2) * N + (j_idx + 0)] += tmp20;
              res[(i_idx + 2) * N + (j_idx + 1)] += tmp21;
              res[(i_idx + 2) * N + (j_idx + 2)] += tmp22;
              res[(i_idx + 2) * N + (j_idx + 3)] += tmp23;

              res[(i_idx + 3) * N + (j_idx + 0)] += tmp30;
              res[(i_idx + 3) * N + (j_idx + 1)] += tmp31;
              res[(i_idx + 3) * N + (j_idx + 2)] += tmp32;
              res[(i_idx + 3) * N + (j_idx + 3)] += tmp33;
            }
          }
        }
      }
    }
  }
}

void matmul_sse(const float* lhs, const float* rhs, float* res, int M, int N, int K) {
  __m128 tmp0, tmp1, tmp2, tmp3;
  __m128 lhs_v, rhs_v;

  // FIXME: only support M%4 = N%4 = 0
  //http://www.plutospin.com/files/OpenMP_reference.pdf
#pragma omp parallel for num_threads(4) private(lhs_v) private(rhs_v) private(tmp0) private(tmp1) private(tmp2) private(tmp3)
  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; j += 4) {
      tmp0 = _mm_setzero_ps();
      tmp1 = _mm_setzero_ps();
      tmp2 = _mm_setzero_ps();
      tmp3 = _mm_setzero_ps();
      for (int k = 0; k < K; ++k) {
        // res[i,j:j+4] = tmp0 = sum(lhs[i,k] * rhs[k,j:j+4])
        // lhs_v = [lhs[i,k], lhs[i,k], lhs[i,k], lhs[i,k]]         ---> mm_load_ps1
        // rhs_v = [rhs[k,j], rhs[k,j+1], rhs[k,j+2], rhs[k,j+3]]   ---> mm_load_ps
        rhs_v = _mm_load_ps(&rhs[k * N + j]);
        lhs_v = _mm_load_ps1(&lhs[(i + 0) * K + k]);
        tmp0 = _mm_fmadd_ps(lhs_v, rhs_v, tmp0);

        lhs_v = _mm_load_ps1(&lhs[(i + 1) * K + k]);
        tmp1 = _mm_fmadd_ps(lhs_v, rhs_v, tmp1);

        lhs_v = _mm_load_ps1(&lhs[(i + 2) * K + k]);
        tmp2 = _mm_fmadd_ps(lhs_v, rhs_v, tmp2);

        lhs_v = _mm_load_ps1(&lhs[(i + 3) * K + k]);
        tmp3 = _mm_fmadd_ps(lhs_v, rhs_v, tmp3);

      }
      _mm_store_ps(&res[(i + 0) * N + j], tmp0);
      _mm_store_ps(&res[(i + 1) * N + j], tmp1);
      _mm_store_ps(&res[(i + 2) * N + j], tmp2);
      _mm_store_ps(&res[(i + 3) * N + j], tmp3);
    }
  }
}

// TODO: unfortunately my machine doesn't support avx512
/* void matmul_avx512(std::shared_ptr<float[]> lhs, std::shared_ptr<float[]> rhs, std::shared_ptr<float[]> res, int M, int N, int K) { */
/*   __m512 tmp0, tmp1, tmp2, tmp3; */
/*   __m512 lhs_v, rhs_v; */

/*   // FIXME: only support M%4 = N%16 = 0 */
/*   for (int i = 0; i < M; i += 4) { */
/*     for (int j = 0; j < N; j += 16) { */
/*       tmp0 = _mm512_setzero_ps(); */
/*       tmp1 = _mm512_setzero_ps(); */
/*       tmp2 = _mm512_setzero_ps(); */
/*       tmp3 = _mm512_setzero_ps(); */
/*       for (int k = 0; k < K; ++k) { */
/*         // res[i,j:j+4] = tmp0 = sum(lhs[i,k] * rhs[k,j:j+4]) */
/*         // lhs_v = [lhs[i,k], lhs[i,k], lhs[i,k], lhs[i,k], ...]         ---> mm_load_ps1 */
/*         // rhs_v = [rhs[k,j], rhs[k,j+1], rhs[k,j+2], rhs[k,j+3], ...]   ---> mm_load_ps */
/*         rhs_v = _mm512_loadu_ps(&rhs[k * N + j]); */
/*         lhs_v = _mm512_set1_ps(lhs[i * K + k]); */
/*         tmp0 = _mm512_fmadd_ps(lhs_v, rhs_v, tmp0); */

/*         lhs_v = _mm512_set1_ps(lhs[(i + 1) * K + k]); */
/*         tmp1 = _mm512_fmadd_ps(lhs_v, rhs_v, tmp1); */

/*         lhs_v = _mm512_set1_ps(lhs[(i + 2) * K + k]); */
/*         tmp2 = _mm512_fmadd_ps(lhs_v, rhs_v, tmp2); */

/*         lhs_v = _mm512_set1_ps(lhs[(i + 3) * K + k]); */
/*         tmp3 = _mm512_fmadd_ps(lhs_v, rhs_v, tmp3); */

/*       } */
/*       _mm512_storeu_ps(&res[(i + 0) * N + j], tmp0); */
/*       _mm512_storeu_ps(&res[(i + 1) * N + j], tmp1); */
/*       _mm512_storeu_ps(&res[(i + 2) * N + j], tmp2); */
/*       _mm512_storeu_ps(&res[(i + 3) * N + j], tmp3); */
/*     } */
/*   } */
/* } */
