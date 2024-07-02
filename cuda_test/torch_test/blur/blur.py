from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

def compile_extension():
    cuda_source = Path("blur.cu").read_text()
    cpp_source = "torch::Tensor blur(torch::Tensor image, int kernel_size);"

    blur_extension = load_inline(
        name = "blur_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["blur"],
        with_cuda = True,
        extra_cuda_cflags = ["-O2"],
        # build_directory = "./cuda_build",
    )

    return blur_extension

def main():
    ext = compile_extension()
    
    x = read_image("./test.jpg").contiguous().cuda()

    y = ext.blur(x, 3)
    write_png(y.cpu(), "output.jpg")

if __name__ == "__main__":
    main()
