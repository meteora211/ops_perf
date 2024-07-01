from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

def compile_extension():
    cuda_source = Path("color2gray.cu").read_text()
    cpp_source = "torch::Tensor color2gray(torch::Tensor image);"

    color2gray_extension = load_inline(
        name = "color2gray_extension",
        cpp_sources = cpp_source,
        cuda_sources = cuda_source,
        functions = ["color2gray"],
        with_cuda = True,
        extra_cuda_cflags = ["-O2"],
        build_directory = "./cuda_build",
    )

    return color2gray_extension

def main():
    ext = compile_extension()
    
    x = read_image("./test.jpg").permute(1, 2, 0).cuda()

    y = ext.color2gray(x)
    write_png(y.permute(2, 0, 1).cpu(), "output.jpg")

if __name__ == "__main__":
    main()
