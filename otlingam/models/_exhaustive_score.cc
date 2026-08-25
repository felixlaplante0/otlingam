#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "_exhaustive_score.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace otlingam {
namespace HWY_NAMESPACE {

double sum_squared_differences_impl(
    const double *values,
    const double *quantiles,
    int size) noexcept {
  const hwy::HWY_NAMESPACE::ScalableTag<double> d;
  auto sum = hwy::HWY_NAMESPACE::Zero(d);
  int i = 0;
  const int lanes = static_cast<int>(hwy::HWY_NAMESPACE::Lanes(d));
  for (; i + lanes <= size; i += lanes) {
    const auto delta = hwy::HWY_NAMESPACE::Sub(
        hwy::HWY_NAMESPACE::LoadU(d, values + i),
        hwy::HWY_NAMESPACE::LoadU(d, quantiles + i));
    sum = hwy::HWY_NAMESPACE::Add(sum, hwy::HWY_NAMESPACE::Mul(delta, delta));
  }
  double result = hwy::HWY_NAMESPACE::ReduceSum(d, sum);
  for (; i < size; ++i) {
    const double delta = values[i] - quantiles[i];
    result += delta * delta;
  }
  return result;
}

}  // namespace HWY_NAMESPACE
}  // namespace otlingam
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace otlingam {
HWY_EXPORT(sum_squared_differences_impl);

double sum_squared_differences(
    const double *values,
    const double *quantiles,
    int size) noexcept {
  return HWY_DYNAMIC_DISPATCH(sum_squared_differences_impl)(
      values, quantiles, size);
}
}  // namespace otlingam
#endif
