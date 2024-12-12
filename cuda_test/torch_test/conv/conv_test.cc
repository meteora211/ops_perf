#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

constexpr int TILE_WIDTH = 16;

constexpr int INTILE = 32;

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1)/ b;
}


std::vector<int64_t> calc_conv_output_shape(const torch::Tensor& input,
                                            const torch::Tensor& weight,
                                            const c10::IntArrayRef stride,  
                                            const c10::IntArrayRef padding) {

  auto input_size = input.sizes();
  auto weight_size = weight.sizes();
  std::vector<int64_t> output_size(input_size.size());

  output_size[0] = input_size[0];
  output_size[1] = weight_size[0];
  for (int i = 2; i < input_size.size(); ++i) {
    output_size[i] = (input_size[i] + 2 * padding[i - 2] - weight_size[i]) / stride[i - 2] + 1;
  }

  return output_size;
}

void conv2dRef(float* input, float* kernel, float* res, 
               const int height, const int width, const int kernel_h, const int kernel_w,
               const int batch_size, const int output_channel, const int input_channel,
               const int pad_h, const int pad_w, const int stride_h, const int stride_w) {
  int res_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int res_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < output_channel; ++oc) {
      float* res_offset = res + b * output_channel * res_h * res_w + oc * res_h * res_w;
      for (int ic = 0; ic < input_channel; ++ic) {

        for (int si = 0; si < height; si+=stride_h) {
          for (int sj = 0; sj < width; sj+=stride_w) {
            float* input_offset = input + b * input_channel * height * width + ic * height * width;
            float* kernel_offset = kernel + oc * input_channel * kernel_h * kernel_w + ic * kernel_h * kernel_w;

            float sum = 0.0;
            for (int ki = 0; ki < kernel_h; ++ki) {
              for (int kj = 0; kj < kernel_w; ++kj) {
                int offset_h = si + ki - pad_h;
                int offset_w = sj + kj - pad_w;
                if (offset_h < 0 || offset_h >= height || offset_w < 0 || offset_w >= width) continue;
                sum += input_offset[offset_h * width + offset_w] * kernel_offset[ki * kernel_w + kj];
              }
            }

            res_offset[si / stride_h * res_w + sj / stride_w] += sum;
          }
        }
      }
    }
  }
}

__global__ void conv2d_basic(float* input, float* kernel, float* res,
                             const int height, const int width, const int kernel_h, const int kernel_w,
                             const int batch_size, const int output_channel, const int input_channel,
                             const int pad_h, const int pad_w, const int stride_h, const int stride_w) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int res_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int res_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < output_channel; ++oc) {
      float* res_offset = res + b * output_channel * res_h * res_w + oc * res_h * res_w;
      for (int ic = 0; ic < input_channel; ++ic) {
        float* input_offset = input + b * input_channel * height * width + ic * height * width;
        float* kernel_offset = kernel + oc * input_channel * kernel_h * kernel_w + ic * kernel_h * kernel_w;
        float sum = 0.0;
        for (int ki = 0; ki < kernel_h; ++ki) {
          for (int kj = 0; kj < kernel_w; ++kj) {
            int offset_h = row * stride_h + ki - pad_h;
            int offset_w = col * stride_w + kj - pad_w;
            if (offset_h < 0 || offset_h >= height || offset_w < 0 || offset_w >= width) continue;
            sum += input_offset[offset_h * width + offset_w] * kernel_offset[ki * kernel_w + kj];
          }
        }

        res_offset[row * res_w + col] += sum;
      }
    }
  }
}

__global__ void conv2d_opt(float* input, float* kernel, float* res,
                             const int height, const int width, const int kernel_h, const int kernel_w,
                             const int batch_size, const int output_channel, const int input_channel,
                             const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                             const int res_h, const int res_w, const int OUTTILE) {
  int row = blockIdx.y * OUTTILE + threadIdx.y;
  int col = blockIdx.x * OUTTILE + threadIdx.x;
  int tile_pad_h = (blockIdx.y == 0 || blockIdx.y == blockDim.y) ? pad_h : 0;
  int tile_pad_w = (blockIdx.x == 0 || blockIdx.x == blockDim.x) ? pad_w : 0;
  __shared__ float inTile[INTILE][INTILE];

  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < output_channel; ++oc) {
      float* res_offset = res + b * output_channel * res_h * res_w + oc * res_h * res_w;
      for (int ic = 0; ic < input_channel; ++ic) {
        float* input_offset = input + b * input_channel * height * width + ic * height * width;
        float* kernel_offset = kernel + oc * input_channel * kernel_h * kernel_w + ic * kernel_h * kernel_w;
        float sum = 0.0;

        if (row >= 0 && row < height && col >= 0 && col < width) {
          inTile[threadIdx.y][threadIdx.x] = input_offset[row * width + col];
        } else {
          inTile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();


        // only activate OUTTILE*OUTTILE threads
        if (threadIdx.x < OUTTILE && threadIdx.y < OUTTILE)  {
          for (int ki = 0; ki < kernel_h; ++ki) {
            for (int kj = 0; kj < kernel_w; ++kj) {
              int tileRow = threadIdx.y * stride_h + ki - tile_pad_h;
              int tileCol = threadIdx.x * stride_w + kj - tile_pad_w;
              if (row < 0 || row >= height || col < 0 || col >= width || tileRow < 0 || tileRow >= INTILE || tileCol < 0 || tileCol >= INTILE) continue;
              sum += inTile[tileRow][tileCol] * kernel_offset[ki * kernel_w + kj];
            }
          }
        }


        res_offset[row * res_w + col] += sum;
        __syncthreads();
      }
    }
  }

}


torch::Tensor conv2d(const torch::Tensor& input,
                   const torch::Tensor& weight,
                   const c10::IntArrayRef stride,
                   const c10::IntArrayRef padding) {
  auto output_size = calc_conv_output_shape(input, weight, stride, padding);
  auto input_dim = input.dim();
  const int input_height = input.size(input_dim - 2);
  const int input_width = input.size(input_dim - 1);
  const int input_channel = input.size(input_dim - 3);
  const int output_channel = weight.size(0);
  const int weight_height = weight.size(2);
  const int weight_width = weight.size(3);
  const int batch_size = input.dim() == 4 ? input.size(0) : 1;
  auto output = torch::zeros(output_size, input.options());
  const int output_height = output.size(input_dim - 2);
  const int output_width = output.size(input_dim - 1);

  // cpu reference
  // conv2dRef(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
  //           input_height, input_width, weight_height, weight_width, 
  //           batch_size, weight.size(0), input_channel,
  //           padding[0], padding[1], stride[0], stride[1]);

  // cuda basic 
  // dim3 thread_per_block(TILE_WIDTH, TILE_WIDTH);
  // dim3 block_per_grid(cdiv(output_size[input_dim - 2], TILE_WIDTH), cdiv(output_size[input_dim - 1], TILE_WIDTH));
  //
  // conv2d_basic<<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
  //           input_height, input_width, weight_height, weight_width, 
  //           batch_size, weight.size(0), input_channel,
  //           padding[0], padding[1], stride[0], stride[1]);

  // share memory
  dim3 thread_per_block(INTILE, INTILE);
  // FIXME: only support height == width
  int  OUTTILE = (INTILE + 2 * padding[0] - weight_height) / stride[0] + 1;
  std::cout << "============ OUTTILE============" << OUTTILE << std::endl;
  dim3 block_per_grid(cdiv(output_size[input_dim - 2], OUTTILE), cdiv(output_size[input_dim - 1], OUTTILE));

  conv2d_opt<<<block_per_grid, thread_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            input_height, input_width, weight_height, weight_width, batch_size,
            output_channel, input_channel, padding[0], padding[1], stride[0], stride[1],
            output_height, output_width, OUTTILE);

  return output;
}
