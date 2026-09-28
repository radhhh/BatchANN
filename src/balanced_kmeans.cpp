// See include/balanced_kmeans.h.
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#include <mkl.h>
#include <omp.h>

#include "balanced_kmeans.h"
#include "ann_exception.h"
#include "cached_io.h"
#include "logger.h"
#include "math_utils.h"
#include "partition.h"
#include "timer.h"
#include "utils.h"

namespace diskann
{
namespace bkmeans
{

static constexpr uint32_t NONE = std::numeric_limits<uint32_t>::max();

void topm_centers_block(const float *data, size_t n, size_t dim, const float *data_l2sq, const float *centers,
                        size_t K, const float *centers_l2sq, uint32_t m, uint32_t *out_idx, float *out_dist,
                        float *scratch)
{
    if (m == 0 || m > K)
        throw diskann::ANNException("topm_centers_block: m must be in [1, K]", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (n == 0)
        return;

    // scratch[i][j] = -2 * <x_i, c_j>; the norms are added per row below.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (MKL_INT)n, (MKL_INT)K, (MKL_INT)dim, -2.0f, data,
                (MKL_INT)dim, centers, (MKL_INT)dim, 0.0f, scratch, (MKL_INT)K);

#pragma omp parallel
    {
        std::vector<uint32_t> idx(K);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < (int64_t)n; i++)
        {
            float *row = scratch + (size_t)i * K;
            const float xi = data_l2sq[i];
            for (size_t j = 0; j < K; j++)
                row[j] += xi + centers_l2sq[j];
            std::iota(idx.begin(), idx.end(), 0u);
            std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), [row](uint32_t a, uint32_t b) {
                return row[a] < row[b] || (row[a] == row[b] && a < b);
            });
            for (uint32_t l = 0; l < m; l++)
            {
                out_idx[(size_t)i * m + l] = idx[l];
                out_dist[(size_t)i * m + l] = std::max(0.0f, row[idx[l]]);
            }
        }
    }
}

namespace
{
struct Proposal
{
    uint32_t c;
    float d;
    uint32_t p;
};
} // namespace

void capped_assign(const uint32_t *cand, const float *cdist, size_t n, uint32_t m, uint32_t K, size_t cap,
                   const float *data, size_t dim, const float *centers, uint32_t *label)
{
    if ((size_t)K * cap < n)
        throw diskann::ANNException("capped_assign: K * cap is smaller than n", -1, __FUNCSIG__, __FILE__, __LINE__);

    std::fill(label, label + n, NONE);
    std::vector<size_t> filled(K, 0);
    std::vector<uint32_t> level(n, 0);
    std::vector<uint32_t> pending(n);
    std::iota(pending.begin(), pending.end(), 0u);
    std::vector<uint32_t> leftover, next;
    std::vector<Proposal> props;

    while (!pending.empty())
    {
        props.resize(pending.size());
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)pending.size(); i++)
        {
            const uint32_t p = pending[i];
            const size_t slot = (size_t)p * m + level[p];
            props[i] = Proposal{cand[slot], cdist[slot], p};
        }
        std::sort(props.begin(), props.end(), [](const Proposal &a, const Proposal &b) {
            if (a.c != b.c)
                return a.c < b.c;
            if (a.d != b.d)
                return a.d < b.d;
            return a.p < b.p;
        });

        next.clear();
        size_t i = 0;
        while (i < props.size())
        {
            const uint32_t c = props[i].c;
            size_t j = i;
            while (j < props.size() && props[j].c == c)
                j++;
            const size_t room = cap > filled[c] ? cap - filled[c] : 0;
            const size_t take = std::min(room, j - i);
            for (size_t k = i; k < i + take; k++)
                label[props[k].p] = c;
            filled[c] += take;
            for (size_t k = i + take; k < j; k++)
            {
                const uint32_t p = props[k].p;
                level[p]++;
                if (level[p] >= m)
                    leftover.push_back(p);
                else
                    next.push_back(p);
            }
            i = j;
        }
        pending.swap(next);
    }

    // Points that were rejected by all their candidates: nearest cluster with room, one by one.
    std::vector<float> d(K);
    for (uint32_t p : leftover)
    {
        const float *x = data + (size_t)p * dim;
        uint32_t best = NONE;
        float best_d = std::numeric_limits<float>::max();
        for (uint32_t c = 0; c < K; c++)
        {
            if (filled[c] >= cap)
                continue;
            const float *cc = centers + (size_t)c * dim;
            float s = 0;
            for (size_t k = 0; k < dim; k++)
            {
                const float t = x[k] - cc[k];
                s += t * t;
            }
            if (s < best_d)
            {
                best_d = s;
                best = c;
            }
        }
        label[p] = best; // K * cap >= n guarantees a cluster with room
        filled[best]++;
    }
}

void train_balanced(const float *sample, size_t n, size_t dim, const Params &p, float *centroids)
{
    const uint32_t K = p.K;
    if (K < 2 || n < K)
        throw diskann::ANNException("train_balanced: need 2 <= K <= n", -1, __FUNCSIG__, __FILE__, __LINE__);
    const uint32_t m = std::min<uint32_t>(std::max<uint32_t>(p.ncand, 1), K);
    const size_t cap = (n + K - 1) / K;
    const size_t block = std::min(p.block, n);

    std::vector<float> sample_l2sq(n), centers_l2sq(K);
    math_utils::compute_vecs_l2sq(sample_l2sq.data(), const_cast<float *>(sample), n, dim);

    std::vector<uint32_t> cand((size_t)n * m), label(n, NONE), new_label(n);
    std::vector<float> cdist((size_t)n * m), scratch((size_t)block * K);
    std::vector<double> sums((size_t)K * dim);
    std::vector<size_t> counts(K);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> pick(0, n - 1);

    for (uint32_t it = 0; it < p.balanced_iters; it++)
    {
        math_utils::compute_vecs_l2sq(centers_l2sq.data(), centroids, K, dim);
        for (size_t s = 0; s < n; s += block)
        {
            const size_t nb = std::min(block, n - s);
            topm_centers_block(sample + s * dim, nb, dim, sample_l2sq.data() + s, centroids, K, centers_l2sq.data(),
                               m, cand.data() + s * m, cdist.data() + s * m, scratch.data());
        }
        capped_assign(cand.data(), cdist.data(), n, m, K, cap, sample, dim, centroids, new_label.data());

        size_t changed = 0;
        for (size_t i = 0; i < n; i++)
            changed += (new_label[i] != label[i]);
        label.swap(new_label);

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n; i++)
        {
            double *s = sums.data() + (size_t)label[i] * dim;
            const float *x = sample + i * dim;
            for (size_t k = 0; k < dim; k++)
                s[k] += x[k];
            counts[label[i]]++;
        }
        size_t empty = 0, largest = 0;
        for (uint32_t c = 0; c < K; c++)
        {
            largest = std::max(largest, counts[c]);
            float *cc = centroids + (size_t)c * dim;
            if (counts[c] > 0)
            {
                for (size_t k = 0; k < dim; k++)
                    cc[k] = (float)(sums[(size_t)c * dim + k] / (double)counts[c]);
            }
            else
            {
                empty++;
                std::memcpy(cc, sample + pick(rng) * dim, dim * sizeof(float));
            }
        }
        diskann::cout << "balanced k-means round " << it << ": changed=" << (double)changed / (double)n
                      << " largest=" << largest << " cap=" << cap << " empty=" << empty << std::endl;
        if (changed < n / 1000)
            break;
    }
}

template <typename T>
void cluster_base_file(const std::string &data_file, const Params &p, std::vector<float> &centroids,
                       std::vector<uint32_t> &labels)
{
    Timer timer;
    size_t N, dim;
    diskann::get_bin_metadata(data_file, N, dim);
    const uint32_t K = p.K;
    if (K < 2 || (size_t)K > N)
        throw diskann::ANNException("cluster_base_file: need 2 <= K <= number of points", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);

    // 1. sample and train
    const double frac = std::min(1.0, (double)p.train_points / (double)N);
    float *sample = nullptr;
    size_t n = 0, sdim = 0;
    gen_random_slice<T>(data_file, frac, sample, n, sdim);
    if (n < K)
        throw diskann::ANNException("cluster_base_file: training sample smaller than K", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    diskann::cout << "Balanced k-means: K=" << K << ", training on " << n << " of " << N << " points" << std::endl;

    centroids.assign((size_t)K * dim, 0.0f);
    kmeans::kmeanspp_selecting_pivots(sample, n, dim, centroids.data(), K);
    if (p.warm_iters > 0)
        kmeans::run_lloyds(sample, n, dim, centroids.data(), K, p.warm_iters, NULL, NULL);
    train_balanced(sample, n, dim, p, centroids.data());
    delete[] sample;
    diskann::cout << timer.elapsed_seconds_for_step("balanced k-means training") << std::endl;

    // 2. label every point of the file by its nearest centroid
    timer.reset();
    labels.resize(N);
    const size_t block = std::min(p.block, N);
    std::unique_ptr<T[]> block_T = std::make_unique<T[]>(block * dim);
    std::unique_ptr<float[]> block_f = std::make_unique<float[]>(block * dim);
    cached_ifstream reader(data_file, 64 * 1024 * 1024);
    uint32_t npts32, dim32;
    reader.read((char *)&npts32, sizeof(uint32_t));
    reader.read((char *)&dim32, sizeof(uint32_t));
    for (size_t start = 0; start < N; start += block)
    {
        const size_t nb = std::min(block, N - start);
        reader.read((char *)block_T.get(), nb * dim * sizeof(T));
        diskann::convert_types<T, float>(block_T.get(), block_f.get(), nb, dim);
        math_utils::compute_closest_centers(block_f.get(), nb, dim, centroids.data(), K, 1, labels.data() + start);
    }

    std::vector<size_t> counts(K, 0);
    for (uint32_t l : labels)
        counts[l]++;
    const auto mm = std::minmax_element(counts.begin(), counts.end());
    diskann::cout << "Cluster sizes: min=" << *mm.first << " mean=" << (double)N / (double)K << " max=" << *mm.second
                  << std::endl;
    diskann::cout << timer.elapsed_seconds_for_step("labeling the base") << std::endl;
}

template DISKANN_DLLEXPORT void cluster_base_file<float>(const std::string &, const Params &, std::vector<float> &,
                                                         std::vector<uint32_t> &);
template DISKANN_DLLEXPORT void cluster_base_file<int8_t>(const std::string &, const Params &, std::vector<float> &,
                                                          std::vector<uint32_t> &);
template DISKANN_DLLEXPORT void cluster_base_file<uint8_t>(const std::string &, const Params &, std::vector<float> &,
                                                           std::vector<uint32_t> &);

} // namespace bkmeans
} // namespace diskann
