#include "_exhaustive_sort.h"

extern "C" void float64_sort(double *values, long long size);

namespace otlingam {

void sort_values(double *values, std::size_t size) {
  float64_sort(values, static_cast<long long>(size));
}

}  // namespace otlingam
