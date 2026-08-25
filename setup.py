"""Build the Highway exhaustive core."""

import sys
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"
WINDOWS = sys.platform == "win32"
COMPILE_ARGS = ["-O3", "-pthread"]
if WINDOWS:
    COMPILE_ARGS += ["-std=c++17", "-DHWY_DISABLE_FUTEX"]


setup(
    options={"build_ext": {"compiler": "mingw32"}} if WINDOWS else {},
    ext_modules=[
        Pybind11Extension(
            "otlingam.models._exhaustive_kernel",
            [
                "otlingam/models/_exhaustive_kernel.cc",
                "otlingam/models/_exhaustive_sort.cc",
                "otlingam/models/_exhaustive_score.cc",
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
            include_dirs=[np.get_include(), str(HIGHWAY), str(ROOT / "otlingam/models")],
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=["-pthread"],
            cxx_std=0 if WINDOWS else 17,
        ),
    ],
)
