//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
// https://arxiv.org/pdf/1804.06826.pdf
#include <cuda.h>
#include <stdio.h>

// 3090ti
#define THREADS_PER_BLOCK 1024
#define THREAD_PER_SM 1536
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define WARP_SIZE 32
#define REPEAT_TIMES 4096
#define L1_SIZE (192 * 1024)      // Ampere L1 size is 192KB
// #define ARRAY_SIZE L1_SIZE / 8   // ARRAY_SIZE has to be less than L1_SIZE
#define CLK_FREQ 1313
// FREQ for 3090Ti https://en.wikipedia.org/wiki/GeForce_30_series
// https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829

// PTX ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html


template<int ARRAY_SIZE>
__global__ void l1_bandwidth_4x32(uint32_t* start_clk, uint32_t* end_clk, float* dest, float* src) {
  uint32_t tid = threadIdx.x;
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;

  // register to avoid compiler optimization
  float sink0 = 0;
  float sink1 = 0;
  float sink2 = 0;
  float sink3 = 0;

  // wrap up
  for (uint32_t i = tid * 4; i < ARRAY_SIZE; i += THREADS_PER_BLOCK * 4) {
    float* ptr = src + i;
    asm volatile ("{\t\n"
                  ".reg .f32 data<4>;\n\t" // create data[4] array with fp32 datatype on register
                  "ld.global.ca.v4.f32 {data0,data1,data2,data3}, [%4];\n\t" // load 4 fp32 data(.v4.f32) from global memory(ld.global). using cache(.ca).
                  "add.f32 %0, data0, %0;\n\t"
                  "add.f32 %1, data1, %1;\n\t"
                  "add.f32 %2, data2, %2;\n\t"
                  "add.f32 %3, data3, %3;\n\t"
                  "}" : "+f"(sink0), "+f"(sink1), "+f"(sink2), "+f"(sink3)  : "l"(ptr) : "memory"
    );
    // + means reading and writing: https://gcc.gnu.org/onlinedocs/gcc/extensions-to-the-c-language-family/how-to-use-inline-assembly-language-in-c-code.html#output-operands
    // f means fp32 reg, l means u64: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  }

  // synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

  // load from l1 cache
  for (uint32_t j = 0; j < REPEAT_TIMES; j++) {
    float* ptr = src + (tid * 4 + (j * WARP_SIZE * 4) % ARRAY_SIZE);
    asm volatile ("{\t\n"
                  ".reg .f32 data<4>;\n\t" // create data[4] array with fp32 datatype on register
                  "ld.global.ca.v4.f32 {data0,data1,data2,data3}, [%4];\n\t" // load 4 fp32 data(.v4.f32) from global memory(ld.global). using cache(.ca).
                  "add.f32 %0, data0, %0;\n\t"
                  "add.f32 %1, data1, %1;\n\t"
                  "add.f32 %2, data2, %2;\n\t"
                  "add.f32 %3, data3, %3;\n\t"
                  "}" : "+f"(sink0), "+f"(sink1), "+f"(sink2), "+f"(sink3)  : "l"(ptr) : "memory"
    );
  }

  // synchronize all threads
	asm volatile ("bar.sync 0;");

	// end timing
	uint32_t end = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");

  // write back
  start_clk[uid] = start;
  end_clk[uid] = end;
  dest[uid] = sink0 + sink1 + sink2 + sink3;
}

template<int ARRAY_SIZE>
__global__ void l1_bandwidth_32(uint32_t* start_clk, uint32_t* end_clk, float* dest, float* src) {
  uint32_t tid = threadIdx.x;
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;

  // register to avoid compiler optimization
  float sink0 = 0;
  float sink1 = 0;
  float sink2 = 0;
  float sink3 = 0;

  // wrap up
  for (uint32_t i = tid * 4; i < ARRAY_SIZE; i += THREADS_PER_BLOCK * 4) {
    float* ptr = src + i;
    asm volatile ("{\t\n"
                  ".reg .f32 data;\n\t" // create data with fp32 datatype on register
                  "ld.global.ca.f32 data, [%1];\n\t" // load fp32 data(.f32) from global memory(ld.global). using cache(.ca).
                  "add.f32 %0, data, %0;\n\t"
                  "}" : "+f"(sink0)  : "l"(ptr) : "memory"
    );
    // + means reading and writing: https://gcc.gnu.org/onlinedocs/gcc/extensions-to-the-c-language-family/how-to-use-inline-assembly-language-in-c-code.html#output-operands
    // f means fp32 reg, l means u64: https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  }

  // synchronize all threads
	asm volatile ("bar.sync 0;");

	// start timing
	uint32_t start = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");

  // load from l1 cache
  for (uint32_t j = 0; j < REPEAT_TIMES; j++) {
    float* ptr = src + (tid * 4 + (j * WARP_SIZE * 4) % ARRAY_SIZE);
    asm volatile ("{\t\n"
                  ".reg .f32 data<4>;\n\t" // create data with fp32 datatype on register
                  "ld.global.ca.f32 data0, [%4 + 0];\n\t" // load fp32 data(.f32) from global memory(ld.global). using cache(.ca).
                  "ld.global.ca.f32 data1, [%4 + 128];\n\t" // load ptr[31]
                  "ld.global.ca.f32 data2, [%4 + 256];\n\t" // load ptr[63]
                  "ld.global.ca.f32 data3, [%4 + 384];\n\t" // load ptr[95]
                  "add.f32 %0, data0, %0;\n\t"
                  "add.f32 %1, data1, %1;\n\t"
                  "add.f32 %2, data2, %2;\n\t"
                  "add.f32 %3, data3, %3;\n\t"
                  "}" : "+f"(sink0), "+f"(sink1), "+f"(sink2), "+f"(sink3)  : "l"(ptr) : "memory"
    );
  }

  // synchronize all threads
	asm volatile ("bar.sync 0;");

	// end timing
	uint32_t end = 0;
	asm volatile ("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");

  // write back
  start_clk[uid] = start;
  end_clk[uid] = end;
  dest[uid] = sink0 + sink1 + sink2 + sink3;
}

template <int ARRAY_SIZE, typename KernelFunc>
void launch_benchmark(KernelFunc func) {

  uint32_t* start_clk = static_cast<uint32_t*>(malloc(TOTAL_THREADS * sizeof(uint32_t)));
  uint32_t* end_clk = static_cast<uint32_t*>(malloc(TOTAL_THREADS * sizeof(uint32_t)));

  float* dest = static_cast<float*>(malloc(TOTAL_THREADS * sizeof(float)));
  float* src = static_cast<float*>(malloc(ARRAY_SIZE * sizeof(float)));

  float* dest_d;
  float* src_d;
  cudaMalloc(&dest_d, TOTAL_THREADS * sizeof(float));
  cudaMalloc(&src_d, ARRAY_SIZE * sizeof(float));

  uint32_t* start_clk_d;
  uint32_t* end_clk_d;
  cudaMalloc(&start_clk_d, TOTAL_THREADS * sizeof(uint32_t));
  cudaMalloc(&end_clk_d, TOTAL_THREADS * sizeof(uint32_t));


  for (int i = 0; i < ARRAY_SIZE; ++i) src[i] = i;

  cudaMemcpy(src_d, src, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

  // func.template operator()<ARRAY_SIZE>(std::forward<Args>(args)...);
  func<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(start_clk_d, end_clk_d, dest_d, src_d);

  cudaMemcpy(start_clk, start_clk_d, TOTAL_THREADS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(end_clk, end_clk_d, TOTAL_THREADS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(dest, dest_d, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost);

  double bw, BW;
  bw = static_cast<double>(REPEAT_TIMES * THREAD_PER_SM * 4 /*float4*/ * sizeof(float) ) / static_cast<double>(end_clk[0] - start_clk[0]);
  BW = bw * CLK_FREQ * 1000000 / 1024 / 1024 / 1024;
  printf("L1 bandwidth = %f (byte/clk/SM), %f (GB/s/SM)\n", bw, BW);
  printf("Total Clk number = %u \n", end_clk[0]- start_clk[0]);

}

int main() {

  launch_benchmark<L1_SIZE / 8>(l1_bandwidth_4x32<L1_SIZE / 8 >);
  launch_benchmark<L1_SIZE / 8>(l1_bandwidth_32<L1_SIZE / 8 >);

}
