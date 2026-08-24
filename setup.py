"""Build the Highway exhaustive core."""

import sys
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"

COMPILE_ARGS = ["/O2", "/std:c++17"] if sys.platform == "win32" else [
    "-O3",
    "-std=c++17",
    "-pthread",
]
LINK_ARGS = [] if sys.platform == "win32" else ["-pthread"]

setup(
    ext_modules=[
        Extension(
            "otlingam.models._exhaustive_kernel",
            [
                "otlingam/models/_exhaustive_kernel.cc",
                "otlingam/models/_exhaustive_sort.cc",
                "highway/hwy/abort.cc",
                "highway/hwy/targets.cc",
                "highway/hwy/contrib/sort/vqsort.cc",
                "highway/hwy/contrib/sort/vqsort_f64a.cc",
                "highway/hwy/contrib/sort/vqsort_have.cc",
                "highway/hwy/aligned_allocator.cc",
                "highway/hwy/contrib/thread_pool/thread_pool.cc",
                "highway/hwy/contrib/thread_pool/topology.cc",
                "highway/hwy/profiler.cc",
                "highway/hwy/timer.cc",
            ],
            include_dirs=[np.get_include(), pybind11.get_include(), str(HIGHWAY)],
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=LINK_ARGS,
            language="c++",
        )
    ]
)
