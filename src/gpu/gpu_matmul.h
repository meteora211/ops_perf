#pragma once
#include "utils.h"
#include "baseline.h"
// #include "gpu_common.h"

// Some useful repo:
// https://github.com/Liu-xiandong/How_to_optimize_in_GPU
// https://github.com/nicolaswilde/cuda-tensorcore-hgemm
// https://github.com/nicolaswilde/cuda-sgemm

double matmul_cublas(const float* lhs, const float* rhs, float* res, int M, int N, int K);

double matmul_cuda_naive(const float* lhs, const float* rhs, float* res, int M, int N, int K);

double matmul_cuda_transpose(const float* lhs, const float* rhs, float* res, int M, int N, int K);

double matmul_cuda_block(const float* lhs, const float* rhs, float* res, int M, int N, int K);
