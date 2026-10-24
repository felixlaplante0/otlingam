"""Build the Highway and djbsort exhaustive core."""

import platform
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

WINDOWS = sys.platform == "win32"
MACHINE = platform.machine().lower()
COMPILE_ARGS = ["-O3", "-pthread"]
LINK_ARGS = ["-pthread"]
if WINDOWS:
    COMPILE_ARGS.append("-DHWY_DISABLE_FUTEX")


def djbsort_library(target, int64_impl, float64_impl, cflags=()):
    """Build one djbsort target with its symbols suffixed by the target name."""
    int64_sort = f"djbsort_int64_{target}"
    return (
        f"djbsort_{target}",
        {
            "sources": [
                f"djbsort/int64/{int64_impl}/sort.c",
                f"djbsort/float64/{float64_impl}/sort.c",
            ],
            "macros": [
                ("int64_sort", int64_sort),
                ("djbsort_int64", int64_sort),
                ("float64_sort", f"djbsort_float64_{target}"),
            ],
            "include_dirs": ["djbsort/cryptoint", "otlingam/models/_djbsort_compat"],
            "cflags": ["-O3", *cflags],
        },
    )


LIBRARIES = [djbsort_library("portable", "portable4", "useint64")]
if MACHINE in {"amd64", "x86_64"}:
    LIBRARIES.append(djbsort_library("avx2", "avx2", "avx2useint64", ["-mavx2"]))
elif MACHINE in {"aarch64", "arm64"}:
    LIBRARIES.append(djbsort_library("neon", "neon", "useint64"))

HIGHWAY_SOURCES = [
    "highway/hwy/abort.cc",
    "highway/hwy/targets.cc",
    "highway/hwy/contrib/sort/vqsort.cc",
    "highway/hwy/contrib/sort/vqsort_f64a.cc",
    "highway/hwy/aligned_allocator.cc",
    "highway/hwy/per_target.cc",
    "highway/hwy/contrib/thread_pool/thread_pool.cc",
    "highway/hwy/contrib/thread_pool/topology.cc",
    "highway/hwy/profiler.cc",
    "highway/hwy/timer.cc",
]
# Highway after 1.4.0 splits these out of the sources above.
HIGHWAY_SOURCES += [
    path
    for path in ("highway/hwy/contrib/sort/vqsort_have.cc", "highway/hwy/os_rng.cc")
    if Path(path).exists()
]

EXTENSION = Pybind11Extension(
    "otlingam.models._exhaustive_kernel",
    ["otlingam/models/_exhaustive_kernel.cpp", *HIGHWAY_SOURCES],
    include_dirs=["highway", "otlingam/models"],
    extra_compile_args=COMPILE_ARGS,
    extra_link_args=LINK_ARGS,
    cxx_std=17,
)
if WINDOWS:
    EXTENSION.extra_compile_args = [
        arg for arg in EXTENSION.extra_compile_args if not arg.startswith("/")
    ]

setup(
    options={"build_clib": {"compiler": "mingw32"}, "build_ext": {"compiler": "mingw32"}}
    if WINDOWS
    else {},
    libraries=LIBRARIES,
    ext_modules=[EXTENSION],
)
