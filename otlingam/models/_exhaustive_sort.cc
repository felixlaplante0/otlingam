#include "_exhaustive_sort.h"

#include "hwy/contrib/sort/order.h"
#include "hwy/contrib/sort/vqsort.h"

namespace otlingam {

void sort_values(double *values, std::size_t size) {
  hwy::VQSort(values, size, hwy::SortAscending());
}

}  // namespace otlingam
