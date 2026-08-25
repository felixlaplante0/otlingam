#include "hwy/detect_targets.h"
#include "hwy/targets.h"

extern "C" void djbsort_float64_portable(double *, long long);

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
extern "C" void djbsort_float64_avx2(double *, long long);
#endif

#if defined(__aarch64__)
extern "C" void djbsort_float64_neon(double *, long long);
#endif

extern "C" void djbsort_float64(double *values, long long size) {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
  if ((hwy::SupportedTargets() & HWY_AVX2) != 0) {
    djbsort_float64_avx2(values, size);
    return;
  }
#elif defined(__aarch64__)
  if ((hwy::SupportedTargets() & HWY_ALL_NEON) != 0) {
    djbsort_float64_neon(values, size);
    return;
  }
#endif
  djbsort_float64_portable(values, size);
}
