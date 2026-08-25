#define CRYPTO_NAMESPACETOP djbsort_int64
#if defined(__aarch64__)
#include "../djbsort-debian/int64/neon/sort.c"
#else
#include "../djbsort-debian/int64/avx2/sort.c"
#endif
