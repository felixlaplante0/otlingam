#if defined(__aarch64__)
#define CRYPTO_NAMESPACETOP djbsort_float64_neon
#define djbsort_int64 djbsort_int64_neon
#include "useint64/sort.c"
#endif
