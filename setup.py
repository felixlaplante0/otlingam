"""Build the djbsort exhaustive core."""

import platform
import sys
import sysconfig
from pathlib import Path

import numpy as np
from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext
from setuptools import setup
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

ROOT = Path(__file__).resolve().parent
IS_ARM = "arm" in f"{platform.machine()} {sysconfig.get_platform()}".lower()

COMPILE_ARGS = [
    "/O2", "/arch:AVX2"
] if sys.platform == "win32" else [
    "-O3", "-pthread", "-mavx2", "-mbmi2", "-mpopcnt",
    f"-iquote{ROOT / 'third_party/djbsort-debian/cryptoint'}",
]
LINK_ARGS = [] if sys.platform == "win32" else ["-pthread"]


class BuildExt(build_ext):
    def build_extensions(self):
        original_sources = {id(extension): extension.sources[:] for extension in self.extensions}
        for extension in self.extensions:
            extension.sources = [
                str(ROOT / source) if not Path(source).is_absolute() else source
                for source in extension.sources
            ]
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
                extension.extra_compile_args = [
                    "-O3", "-pthread",
                    f"-iquote{ROOT / 'third_party/djbsort-debian/cryptoint'}",
                ]
                if not IS_ARM:
                    extension.extra_compile_args[1:1] = ["-mavx2", "-mbmi2", "-mpopcnt"]
                extension.extra_link_args = ["-pthread"]
        try:
            super().build_extensions()
        finally:
            for extension in self.extensions:
                extension.sources = original_sources[id(extension)]
setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Pybind11Extension(
            "otlingam.models._exhaustive_kernel",
            [
                "otlingam/models/_exhaustive_kernel.cc",
                "otlingam/models/_exhaustive_sort.cc",
                "third_party/djbsort/int64_avx2_wrapper.c",
                "third_party/djbsort/float64_avx2_wrapper.c",
            ],
            include_dirs=[
                np.get_include(),
                str(ROOT / "third_party/djbsort_compat"),
            ],
            extra_compile_args=COMPILE_ARGS,
            extra_link_args=LINK_ARGS,
            cxx_std=17,
        ),
    ],
)
