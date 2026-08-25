#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
#pragma GCC target("avx2")
#define djbsort_int64 djbsort_int64_avx2
#define CRYPTO_NAMESPACETOP djbsort_float64_avx2
#include "avx2useint64/sort.c"
#pragma GCC reset_options
#endif
