"""Build the Highway and djbsort exhaustive core."""

import sys
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"
DJBSORT = ROOT / "djbsort"
WINDOWS = sys.platform == "win32"
COMPILE_ARGS = ["-O3", "-pthread"]
if WINDOWS:
    COMPILE_ARGS += ["-std=c++17", "-DHWY_DISABLE_FUTEX"]

EXTENSION = Pybind11Extension(
    "otlingam.models._exhaustive_kernel",
    [
        "otlingam/models/_exhaustive_kernel.cc",
        "otlingam/models/_exhaustive_sort.cc",
        "otlingam/models/_exhaustive_score.cc",
        "otlingam/models/_djbsort_dispatch.cc",
        "otlingam/models/_djbsort_int64_portable.c",
        "otlingam/models/_djbsort_float64_portable.c",
        "otlingam/models/_djbsort_int64_avx2.c",
        "otlingam/models/_djbsort_float64_avx2.c",
        "otlingam/models/_djbsort_int64_neon.c",
        "otlingam/models/_djbsort_float64_neon.c",
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
    include_dirs=[
        np.get_include(),
        str(HIGHWAY),
        str(DJBSORT / "int64"),
        str(DJBSORT / "float64"),
        str(ROOT / "otlingam/models"),
        str(ROOT / "otlingam/models/_djbsort_compat"),
    ],
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=["-pthread"],
    cxx_std=0 if WINDOWS else 17,
)
if WINDOWS:
    EXTENSION.extra_compile_args = [
        arg for arg in EXTENSION.extra_compile_args if not arg.startswith("/")
    ]


class BuildExt(build_ext):
    """Keep C-only djbsort sources out of the C++ standard flag."""

    def build_extensions(self):
        compile_source = self.compiler._compile

        def compile_without_cpp_standard(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ):
            if src.endswith(".c"):
                extra_postargs = [
                    arg
                    for arg in extra_postargs
                    if not arg.startswith(("-std=c++", "-std=gnu++"))
                ]
            return compile_source(
                obj, src, ext, cc_args, extra_postargs, pp_opts
            )

        self.compiler._compile = compile_without_cpp_standard
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = compile_source


setup(
    options={"build_ext": {"compiler": "mingw32"}} if WINDOWS else {},
    cmdclass={"build_ext": BuildExt},
    ext_modules=[EXTENSION],
)
