Highway
=======

OTLiNGAM uses `Google Highway <https://github.com/google/highway>`__ for the
sorting kernel in the exhaustive estimator.

Vectorized sorting
------------------

The exhaustive estimator evaluates many residual arrays and sorts each one
before comparing it with Gaussian quantiles. Highway's ``vqsort`` dispatches a
portable SIMD implementation for the available CPU, with a scalar fallback
when needed.

Where the speedup matters
-------------------------

Sorting is the dominant low-level operation in the empirical Wasserstein score.
The Cholesky solve and residual construction stay simple and scalar because
the small per-mask systems and irregular parent masks leave less contiguous
work for SIMD. This keeps the implementation compact while accelerating the
hot sorting path.

The exhaustive estimator also uses OpenMP to evaluate independent candidate
masks in parallel. Its exponential state space still limits it to smaller
systems. Use ``GreedyOTLiNGAM`` when dimension is the main constraint.

Highway is a separate open-source project. See its
`repository <https://github.com/google/highway>`__ for implementation details
and supported targets.
