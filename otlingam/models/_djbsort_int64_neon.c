#if defined(__aarch64__)
#define CRYPTO_NAMESPACETOP djbsort_int64_neon
#include "neon/sort.c"
#endif
