#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools
import glob
from os import path
import torch
from torch.utils.cpp_extension import CppExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "yolox", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "yolox._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


with open("yolox/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="yolox",
    version=version,
    author="basedet team",
    python_requires=">=3.6",
    long_description=long_description,
    ext_modules=get_extensions(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_namespace_packages(),
    install_requires=[
            "torch>=1.7",
            "opencv_python",
            "cython==0.29.28",
            "loguru",
            "scikit-image",
            "tqdm",
            "torchvision>=0.10.0",
            "Pillow",
            "thop",
            "ninja",
            "tabulate",
            "tensorboard",
            "lap",
            "motmetrics",
            "filterpy",
            "h5py",
            "cython_bbox==0.1.3",
            "onnx==1.8.1",
            "onnxruntime==1.8.0",
            "onnx-simplifier==0.3.5",
        ]
)

