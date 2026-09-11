#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"

namespace py = pybind11;

namespace otlingam {

void sort_values(double *values, size_t size) noexcept;
double sum_squared_differences(
    const double *values,
    const double *quantiles,
    int size);

int popcount(int mask) noexcept {
    int result = 0;
    while (mask != 0) {
        result += mask & 1;
        mask >>= 1;
    }
    return result;
}

int build_system(
    const double *cov,
    int d,
    int target,
    int mask,
    double *A,
    int *parents) noexcept {
    int k = 0;
    for (int i = 0; i < d; ++i) {
        if ((mask >> i) & 1) {
            const int kk = k * (k + 1) / 2;
            for (int j = 0; j < k; ++j) {
                A[kk + j] = cov[i * d + parents[j]];
            }
            A[kk + k] = cov[i * d + i];
            parents[k++] = i;
        }
    }

    const int kk = k * (k + 1) / 2;
    for (int j = 0; j < k; ++j) {
        A[kk + j] = cov[target * d + parents[j]];
    }
    A[kk + k] = cov[target * d + target];
    return k;
}

double cholesky(double *A, int k) noexcept {
    for (int i = 0; i <= k; ++i) {
        const int ii = i * (i + 1) / 2;
        for (int j = 0; j < i; ++j) {
            const int ij = ii + j;
            for (int l = 0; l < j; ++l) {
                A[ij] -= A[ii + l] * A[j * (j + 1) / 2 + l];
            }
            A[ij] /= A[j * (j + 1) / 2 + j];
        }
        const int ij = ii + i;
        for (int l = 0; l < i; ++l) {
            A[ij] -= A[ii + l] * A[ii + l];
        }
        if (A[ij] <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        A[ij] = std::sqrt(A[ij]);
    }
    return A[k * (k + 3) / 2] * A[k * (k + 3) / 2];
}

void solve_coefficients(const double *A, int k, double *coef) noexcept {
    for (int i = 0; i < k; ++i) {
        coef[i] = A[k * (k + 1) / 2 + i];
    }
    for (int i = k - 1; i >= 0; --i) {
        for (int j = i + 1; j < k; ++j) {
            coef[i] -= A[j * (j + 1) / 2 + i] * coef[j];
        }
        coef[i] /= A[i * (i + 1) / 2 + i];
    }
}

void compute_residuals(
    const double *X,
    int n,
    int d,
    int target,
    int mask,
    const double *coef,
    double *residuals) noexcept {
    for (int i = 0; i < n; ++i) {
        double value = X[i * d + target];
        int parent = 0;
        for (int j = 0; j < d; ++j) {
            if ((mask >> j) & 1) {
                value -= coef[parent++] * X[i * d + j];
            }
        }
        residuals[i] = value;
    }
}

double score(
    const double *X,
    const double *cov,
    const double *quantiles,
    int n,
    int d,
    int target,
    int mask) noexcept {
    double A[528];
    double coef[31];
    int parents[31];
    std::vector<double> residuals(n);
    const int k = build_system(cov, d, target, mask, A, parents);
    const double rss = cholesky(A, k);
    if (std::isnan(rss)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    solve_coefficients(A, k, coef);
    compute_residuals(X, n, d, target, mask, coef, residuals.data());
    const double scale = std::sqrt(rss / n);
    for (double &value : residuals) {
        value /= scale;
    }
    sort_values(residuals.data(), residuals.size());
    return sum_squared_differences(residuals.data(), quantiles, n) / n;
}

void process_mask(
    const double *X,
    const double *cov,
    const double *quantiles,
    float *scores,
    std::int32_t *sinks,
    int mask,
    int n,
    int d) noexcept {
    int bits = mask;
    int sink = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    int best_sink = -1;
    while (bits != 0) {
        if (bits & 1) {
            const int previous = mask ^ (1 << sink);
            const float candidate = scores[previous] + static_cast<float>(
                score(X, cov, quantiles, n, d, sink, previous));
            if (candidate > best_score) {
                best_score = candidate;
                best_sink = sink;
            }
        }
        bits >>= 1;
        ++sink;
    }
    scores[mask] = best_score;
    sinks[mask] = best_sink;
}

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::tuple sink_dp(
    const Array &X,
    const Array &cov,
    const Array &quantiles,
    int d,
    std::size_t n_jobs) {
    const int n_states = 1 << d;
    const int n = static_cast<int>(X.shape(0));
    py::array_t<std::int32_t> masks(n_states - 1);
    py::array_t<std::int32_t> offsets(d + 2);
    py::array_t<std::int32_t> counts(d + 1);
    py::array_t<std::int32_t> positions(d + 2);
    py::array_t<float> scores(n_states);
    py::array_t<std::int32_t> sinks(n_states);

    auto *mask_data = masks.mutable_data();
    auto *offset_data = offsets.mutable_data();
    auto *count_data = counts.mutable_data();
    auto *position_data = positions.mutable_data();
    auto *score_data = scores.mutable_data();
    auto *sink_data = sinks.mutable_data();
    std::fill(offset_data, offset_data + d + 2, 0);
    std::fill(count_data, count_data + d + 1, 0);
    std::fill(score_data, score_data + n_states, 0.0F);
    std::fill(sink_data, sink_data + n_states, -1);

    for (int mask = 1; mask < n_states; ++mask) {
        ++count_data[popcount(mask)];
    }
    for (int size = 1; size <= d; ++size) {
        offset_data[size + 1] = offset_data[size] + count_data[size];
    }
    std::copy(offset_data, offset_data + d + 2, position_data);
    for (int mask = 1; mask < n_states; ++mask) {
        const int size = popcount(mask);
        mask_data[position_data[size]++] = mask;
    }

    const auto X_view = X.request();
    const auto cov_view = cov.request();
    const auto quantile_view = quantiles.request();
    const auto *X_data = static_cast<const double *>(X_view.ptr);
    const auto *cov_data = static_cast<const double *>(cov_view.ptr);
    const auto *quantile_data = static_cast<const double *>(quantile_view.ptr);

    hwy::AlignedUniquePtr<hwy::ThreadPool> pool;
    n_jobs = n_jobs == 0 ? 1 + hwy::ThreadPool::MaxThreads() : n_jobs;
    if (n_jobs > 1) {
        pool = hwy::MakeUniqueAligned<hwy::ThreadPool>(n_jobs - 1);
        if (!pool) {
            throw std::bad_alloc();
        }
        pool->SetWaitMode(hwy::PoolWaitMode::kSpin);
    }

    {
        py::gil_scoped_release release;
        for (int size = 1; size <= d; ++size) {
            const int start = offset_data[size];
            const int end = offset_data[size + 1];
            const auto run = [&](const int begin, const int finish) {
                for (int index = begin; index < finish; ++index) {
                    process_mask(
                        X_data,
                        cov_data,
                        quantile_data,
                        score_data,
                        sink_data,
                        mask_data[index],
                        n,
                        d);
                }
            };

            if (!pool || end - start <= 1) {
                run(start, end);
                continue;
            }

            const std::size_t count = static_cast<std::size_t>(end - start);
            auto &thread_pool = *pool;
            const std::size_t num_tasks = std::min(
                count, 4 * thread_pool.NumWorkers());
            const std::size_t items_per_task =
                (count + num_tasks - 1) / num_tasks;
            thread_pool.Run(0, num_tasks, [&](const std::uint64_t task, std::size_t) {
                const std::size_t begin = static_cast<std::size_t>(task) * items_per_task;
                const std::size_t finish = std::min(begin + items_per_task, count);
                run(start + static_cast<int>(begin), start + static_cast<int>(finish));
            });
        }
    }
    return py::make_tuple(std::move(sinks), score_data[n_states - 1]);
}

}  // namespace otlingam

PYBIND11_MODULE(_exhaustive_kernel, module) {
    module.def("_sink_dp", &otlingam::sink_dp);
}
