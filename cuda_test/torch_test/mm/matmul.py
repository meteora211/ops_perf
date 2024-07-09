from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

def compile_extension():
    cuda_source = Path("matmul.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor lhs, torch::Tensor rhs);"

    mm_extension = load_inline(
        name = "mm_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["matmul"],
        with_cuda = True,
        extra_cuda_cflags = ["-O2"],
        # build_directory = "./cuda_build",
    )

    return mm_extension

def main():
    ext = compile_extension()
    
    a = torch.randn(1024, 1024).float().cuda()
    b = torch.randn(1024, 1024).float().cuda()

    y = ext.matmul(a, b)
    y_ref = a@b
    print((y - y_ref).abs().max())

if __name__ == "__main__":
    main()
