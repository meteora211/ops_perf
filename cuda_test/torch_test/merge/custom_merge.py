from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import copy

def merge(lhs, rhs):
    cat_tensor = torch.cat((lhs, rhs))
    sorted, _ = torch.sort(cat_tensor)
    return sorted

def compile_extension():
    # cuda_source = Path("conv.cu").read_text()
    cpp_source = "torch::Tensor merge(const torch::Tensor& lhs, const torch::Tensor& rhs);"
    # cpp_source = Path("conv_test.cc").read_text()
    cuda_source = Path("merge_test.cc").read_text()

    merge_extension = load_inline(
        name = "merge_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["merge"],
        with_cuda = True,
        extra_cflags = ["-O0 -g"],
        extra_cuda_cflags = ["-O2"],
        build_directory = "./merge_build",
    )

    return merge_extension

def main():
    ext = compile_extension()
    
    x1, _ = torch.sort(torch.rand(1000).float().cuda())
    x2, _ = torch.sort(torch.rand(2000).float().cuda())


    y_ref = merge(x1, x2)
    print(y_ref)
    import pdb; pdb.set_trace()

    y = ext.merge(x1, x2)
    print(y)
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    main()
