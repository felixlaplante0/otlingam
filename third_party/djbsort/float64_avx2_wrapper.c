#define CRYPTO_NAMESPACETOP float64_sort
#if defined(__aarch64__)
#include "../djbsort-debian/float64/useint64/sort.c"
#else
#include "../djbsort-debian/float64/avx2useint64/sort.c"
#endif
