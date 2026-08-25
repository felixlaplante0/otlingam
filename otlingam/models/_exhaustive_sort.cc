#include "_exhaustive_sort.h"

extern "C" void djbsort_float64(double *, long long);

namespace otlingam {

void sort_values(double *values, std::size_t size) {
  djbsort_float64(values, static_cast<long long>(size));
}

}  // namespace otlingam
