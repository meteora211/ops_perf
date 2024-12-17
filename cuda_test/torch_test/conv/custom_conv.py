from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

def compile_extension():
    # cuda_source = Path("conv.cu").read_text()
    cpp_source = "torch::Tensor conv2d(const torch::Tensor& input, const torch::Tensor& weight, const c10::IntArrayRef stride, const c10::IntArrayRef padding);"
    # cpp_source = Path("conv_test.cc").read_text()
    cuda_source = Path("conv_test.cc").read_text()

    color2gray_extension = load_inline(
        name = "conv_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["conv2d"],
        with_cuda = True,
        extra_cflags = ["-O0 -g"],
        extra_cuda_cflags = ["-O2"],
        build_directory = "./conv_build",
    )

    return color2gray_extension

def main():
    ext = compile_extension()
    
    x = torch.rand(3, 3, 300, 300).float().cuda()
    w = torch.rand(1, 3, 4, 4).float().cuda()

    stride = [2,2]
    pad = [0,0]
    y = ext.conv2d(x, w, stride, pad)
    print(y)
    y_ref = torch.nn.functional.conv2d(x, w, padding = pad, stride = stride)
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    main()
