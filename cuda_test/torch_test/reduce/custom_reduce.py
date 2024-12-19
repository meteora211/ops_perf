from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import copy

def compile_extension():
    # cuda_source = Path("conv.cu").read_text()
    cpp_source = "torch::Tensor reduce(const torch::Tensor& input);"
    # cpp_source = Path("conv_test.cc").read_text()
    cuda_source = Path("reduce_test.cc").read_text()

    reduce_extension = load_inline(
        name = "reduce_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["reduce"],
        with_cuda = True,
        extra_cflags = ["-O0 -g"],
        extra_cuda_cflags = ["-O2"],
        build_directory = "./reduce_build",
    )

    return reduce_extension

def main():
    ext = compile_extension()
    
    # x = torch.tensor([1,2,3,4,5,6,7,8,9,10]).float().cuda()
    # x = torch.tensor([1,10,100,1e3,1e4,1e5,1e6,1e7,1e8,1e9]).float().cuda()
    x = torch.rand(300000).float().cuda()
    # x_clone = copy.deepcopy(x)
    x_clone = x.clone() 

    y_ref = x_clone.sum()
    print(x_clone.sum())
    print(y_ref)

    y = ext.reduce(x)
    print(y)
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    main()
