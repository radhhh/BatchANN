// Balanced k-means for the locality-aware disk layout (RESEARCH_PLAN.md, contribution 4).
//
// Training runs on a random sample of the base with a size cap per cluster, so no cluster
// can swallow a dense region. The whole base is then labeled by plain nearest centroid, with
// no cap, because queries are assigned the same way and the layout should mirror how
// batches form. Built on top of the plain k-means in math_utils.h, which is not changed.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "windows_customizations.h"

namespace diskann
{
namespace bkmeans
{

struct Params
{
    uint32_t K = 0;                    // number of clusters
    size_t train_points = 2000000;     // training sample size: min(N, train_points)
    uint32_t warm_iters = 3;           // plain Lloyd rounds before the balanced ones
    uint32_t balanced_iters = 10;      // balanced rounds (stops early when < 0.1% of labels change)
    uint32_t ncand = 8;                // candidate centers kept per point in a balanced round
    size_t block = 65536;              // points per block when a distance table is computed
};

// For every row of `data` (n x dim, row-major float), the m nearest of the K `centers`,
// sorted by distance. data_l2sq / centers_l2sq hold the squared norms (compute_vecs_l2sq).
// out_idx and out_dist are n*m; scratch is n*K floats of working memory.
DISKANN_DLLEXPORT void topm_centers_block(const float *data, size_t n, size_t dim, const float *data_l2sq,
                                          const float *centers, size_t K, const float *centers_l2sq, uint32_t m,
                                          uint32_t *out_idx, float *out_dist, float *scratch);

// Capacity-constrained assignment. cand/cdist (n x m) list each point's m nearest centers.
// Round by round every unassigned point proposes to its next-best candidate and each cluster
// keeps the closest proposals up to `cap`. Points that exhaust their candidates get the
// nearest cluster that still has room (needs data and centers). label receives n entries.
DISKANN_DLLEXPORT void capped_assign(const uint32_t *cand, const float *cdist, size_t n, uint32_t m, uint32_t K,
                                     size_t cap, const float *data, size_t dim, const float *centers,
                                     uint32_t *label);

// Balanced rounds on `sample` (n x dim). centroids (K x dim) must hold the starting centers
// and receives the result. Cap is ceil(n / K).
DISKANN_DLLEXPORT void train_balanced(const float *sample, size_t n, size_t dim, const Params &p, float *centroids);

// Whole pipeline on a .bin file of T: sample, k-means++ start, plain warm-up rounds,
// balanced rounds, then a nearest-centroid pass over every point of the file.
// centroids: K*dim floats row-major. labels: one entry per point, in file order.
template <typename T>
DISKANN_DLLEXPORT void cluster_base_file(const std::string &data_file, const Params &p, std::vector<float> &centroids,
                                         std::vector<uint32_t> &labels);

} // namespace bkmeans
} // namespace diskann
