"""Build the Highway exhaustive core."""

import platform
import sys
import sysconfig
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

ROOT = Path(__file__).resolve().parent
HIGHWAY = ROOT / "highway"
IS_ARM = "arm" in f"{platform.machine()} {sysconfig.get_platform()}".lower()

COMPILE_ARGS = ["/O2"] if sys.platform == "win32" else [
    "-O3",
    "-pthread",
]
LINK_ARGS = [] if sys.platform == "win32" else ["-pthread"]


class BuildExt(build_ext):
    """Use MinGW on Windows."""

    def build_extensions(self):
        if sys.platform == "win32" and self.compiler.compiler_type != "mingw32":
            self.compiler = new_compiler(compiler="mingw32")
            customize_compiler(self.compiler)
            python_include = sysconfig.get_config_var("INCLUDEPY")
            python_lib = Path(sysconfig.get_config_var("prefix")) / "libs"
            for extension in self.extensions:
                if python_include and python_include not in extension.include_dirs:
                    extension.include_dirs.append(python_include)
                if str(python_lib) not in extension.library_dirs:
                    extension.library_dirs.append(str(python_lib))
        if self.compiler.compiler_type == "mingw32":
            for extension in self.extensions:
                extension.extra_compile_args = ["-O3", "-pthread", "-DHWY_DISABLE_FUTEX"]
                if not IS_ARM:
                    extension.extra_compile_args[1:1] = ["-mavx2", "-mbmi2", "-mpopcnt"]
                extension.extra_link_args = ["-pthread"]
        super().build_extensions()


setup(
    cmdclass={"build_ext": BuildExt},
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
            include_dirs=[np.get_include(), str(HIGHWAY)],
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=LINK_ARGS,
            cxx_std=17,
        ),
    ],
)
