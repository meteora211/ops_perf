//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"
// https://arxiv.org/pdf/1804.06826.pdf
#include <cuda.h>
#include <stdio.h>

// 3090ti
#define WARP_SIZE 32
#define REPEAT_TIMES 256
#define L1_SIZE (192 * 1024)      // Ampere L1 size is 192KB
// #define ARRAY_SIZE L1_SIZE / 8   // ARRAY_SIZE has to be less than L1_SIZE
#define CLK_FREQ 1313
// FREQ for 3090Ti https://en.wikipedia.org/wiki/GeForce_30_series
// https://www.techpowerup.com/gpu-specs/geforce-rtx-3090-ti.c3829
#define L2_BANKS_PER_MEM_CHANNEL 2 
#define L2_BANK_WIDTH_in_BYTE 32

// PTX ref: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

__global__ void l2_bandwidth_4x32(uint32_t* start_clk, uint32_t* end_clk, float* dest, float* src, unsigned ARRAY_SIZE) {
  uint32_t tid = threadIdx.x;
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;

  // register to avoid compiler optimization
  float sink0 = 0;
  float sink1 = 0;
  float sink2 = 0;
  float sink3 = 0;

  // wrap up
  for (uint32_t i = uid; i < ARRAY_SIZE; i += blockDim.x * gridDim.x) {
    float* ptr = src + i;
    asm volatile ("{\t\n"
                  ".reg .f32 data;\n\t" // create data with fp32 datatype on register
                  "ld.global.cg.f32 data, [%1];\n\t" // load fp32 data(.f32) from global memory(ld.global). using cache at global level(.cg, l2 and below not l1).
                  "add.f32 %0, data, %0;\n\t"
                  "}" : "+f"(sink0) : "l"(ptr) : "memory"
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
                  "ld.global.cg.v4.f32 {data0,data1,data2,data3}, [%4];\n\t" // load 4 fp32 data(.v4.f32) from global memory(ld.global). using global cache(.cg).
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

template <typename KernelFunc>
void launch_benchmark(KernelFunc func, unsigned ARRAY_SIZE, unsigned TOTAL_THREADS, unsigned BLOCKS_NUM, unsigned THREADS_PER_BLOCK, unsigned THREADS_PER_SM) {

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
  func<<<BLOCKS_NUM,THREADS_PER_BLOCK>>>(start_clk_d, end_clk_d, dest_d, src_d, ARRAY_SIZE);

  cudaMemcpy(start_clk, start_clk_d, TOTAL_THREADS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(end_clk, end_clk_d, TOTAL_THREADS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(dest, dest_d, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost);

  double bw, BW;
  bw = static_cast<double>(REPEAT_TIMES * TOTAL_THREADS * 4 /*float4*/ * sizeof(float) ) / static_cast<double>(end_clk[0] - start_clk[0]);
  BW = bw * CLK_FREQ * 1000000 / 1024 / 1024 / 1024;
  printf("L2 bandwidth = %f (byte/clk), %f (GB/s)\n", bw, BW);
  printf("Total Clk number = %u \n", end_clk[0]- start_clk[0]);

}

int main() {

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  size_t L2_SIZE = props.l2CacheSize;
  unsigned MEM_BITWIDTH = props.memoryBusWidth;

  printf("L2 SIZE: %d, memory bitwidth: %d\n", L2_SIZE, MEM_BITWIDTH);

  unsigned THREADS_PER_BLOCK = props.maxThreadsPerBlock;
  unsigned SM_NUMBER = props.multiProcessorCount;
  unsigned BLOCKS_PER_SM =
      props.maxThreadsPerMultiProcessor / props.maxThreadsPerBlock;
  unsigned THREADS_PER_SM = BLOCKS_PER_SM * THREADS_PER_BLOCK;
  unsigned BLOCKS_NUM = BLOCKS_PER_SM * SM_NUMBER;
  unsigned TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;
  unsigned ARRAY_SIZE = TOTAL_THREADS * 4 + REPEAT_TIMES * WARP_SIZE * 4;
  launch_benchmark(l2_bandwidth_4x32, ARRAY_SIZE, TOTAL_THREADS, BLOCKS_NUM, THREADS_PER_BLOCK, THREADS_PER_SM);

  // 3090Ti using GDDR6x
  double max_bw = static_cast<double>((MEM_BITWIDTH / 16) /*channels*/ * L2_BANKS_PER_MEM_CHANNEL * L2_BANK_WIDTH_in_BYTE); // I'm not quite understand the number
  double MAX_BW = max_bw * CLK_FREQ * 1000000 / 1024 / 1024 / 1024;
  printf("Max Theortical L2 bandwidth = %f (byte/clk), %f (GB/s)\n", max_bw, MAX_BW);

}
