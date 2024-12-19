from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import copy

def scan(x, exclusive=False):
    res = torch.zeros_like(x)
    res[0] = 0 if exclusive else x[0]
    for i in range(1,x.size()[0]):
        v = x[i - 1] if exclusive else x[i]
        res[i] = res[i - 1] + v
    return res


def compile_extension():
    # cuda_source = Path("conv.cu").read_text()
    cpp_source = "torch::Tensor scan(const torch::Tensor& input, bool exclusive=true);"
    # cpp_source = Path("conv_test.cc").read_text()
    cuda_source = Path("scan_test.cc").read_text()

    scan_extension = load_inline(
        name = "scan_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["scan"],
        with_cuda = True,
        extra_cflags = ["-O0 -g"],
        extra_cuda_cflags = ["-O2"],
        build_directory = "./scan_build",
    )

    return scan_extension

def main():
    ext = compile_extension()
    
    x = torch.rand(300).float().cuda()

    exclusive = True
    y_ref = scan(x, exclusive)
    print(y_ref)

    y = ext.scan(x, exclusive)
    print(y)
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    main()
