// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <cstdio>
#include <random>
#include <vector>

#include "balanced_kmeans.h"
#include "math_utils.h"
#include "utils.h"

namespace
{
// K well-separated centers with Gaussian noise around them.
std::vector<float> planted(size_t n, size_t dim, uint32_t K, std::mt19937 &rng)
{
    std::normal_distribution<float> noise(0.0f, 0.3f);
    std::uniform_real_distribution<float> spread(-10.0f, 10.0f);
    std::vector<float> centers((size_t)K * dim), data(n * dim);
    for (auto &c : centers)
        c = spread(rng);
    for (size_t i = 0; i < n; i++)
    {
        const size_t c = i % K;
        for (size_t k = 0; k < dim; k++)
            data[i * dim + k] = centers[c * dim + k] + noise(rng);
    }
    return data;
}

float sqdist(const float *a, const float *b, size_t dim)
{
    float s = 0;
    for (size_t k = 0; k < dim; k++)
        s += (a[k] - b[k]) * (a[k] - b[k]);
    return s;
}
} // namespace

BOOST_AUTO_TEST_SUITE(balanced_kmeans_tests)

BOOST_AUTO_TEST_CASE(topm_matches_brute_force)
{
    std::mt19937 rng(7);
    const size_t n = 300, dim = 16, K = 40;
    const uint32_t m = 5;
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> data(n * dim), centers(K * dim);
    for (auto &x : data)
        x = u(rng);
    for (auto &x : centers)
        x = u(rng);
    std::vector<float> dl2(n), cl2(K), scratch(n * K), out_dist(n * m);
    std::vector<uint32_t> out_idx(n * m);
    math_utils::compute_vecs_l2sq(dl2.data(), data.data(), n, dim);
    math_utils::compute_vecs_l2sq(cl2.data(), centers.data(), K, dim);
    diskann::bkmeans::topm_centers_block(data.data(), n, dim, dl2.data(), centers.data(), K, cl2.data(), m,
                                         out_idx.data(), out_dist.data(), scratch.data());
    for (size_t i = 0; i < n; i++)
    {
        std::vector<std::pair<float, uint32_t>> all;
        for (uint32_t c = 0; c < K; c++)
            all.push_back({sqdist(data.data() + i * dim, centers.data() + c * dim, dim), c});
        std::sort(all.begin(), all.end());
        for (uint32_t l = 0; l < m; l++)
        {
            BOOST_TEST(out_idx[i * m + l] == all[l].second);
            BOOST_TEST(std::abs(out_dist[i * m + l] - all[l].first) <= 1e-3f * (1.0f + all[l].first));
        }
    }
}

BOOST_AUTO_TEST_CASE(capped_assign_respects_cap)
{
    std::mt19937 rng(11);
    const size_t n = 1000, dim = 8;
    const uint32_t K = 10, m = 4;
    const size_t cap = 100; // exactly n / K, so every cluster must be full
    std::vector<float> data = planted(n, dim, 3, rng); // 3 dense blobs: far from balanced
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    std::vector<float> centers(K * dim);
    for (auto &x : centers)
        x = u(rng);
    std::vector<float> dl2(n), cl2(K), scratch(n * K), cdist(n * m);
    std::vector<uint32_t> cand(n * m), label(n);
    math_utils::compute_vecs_l2sq(dl2.data(), data.data(), n, dim);
    math_utils::compute_vecs_l2sq(cl2.data(), centers.data(), K, dim);
    diskann::bkmeans::topm_centers_block(data.data(), n, dim, dl2.data(), centers.data(), K, cl2.data(), m,
                                         cand.data(), cdist.data(), scratch.data());
    diskann::bkmeans::capped_assign(cand.data(), cdist.data(), n, m, K, cap, data.data(), dim, centers.data(),
                                    label.data());
    std::vector<size_t> counts(K, 0);
    for (size_t i = 0; i < n; i++)
    {
        BOOST_TEST(label[i] < K);
        counts[label[i]]++;
    }
    for (uint32_t c = 0; c < K; c++)
        BOOST_TEST(counts[c] == cap);
}

BOOST_AUTO_TEST_CASE(train_balanced_keeps_cap_and_moves_centers)
{
    std::mt19937 rng(3);
    const size_t n = 4000, dim = 8;
    const uint32_t K = 16;
    std::vector<float> data = planted(n, dim, K, rng);
    std::vector<float> centroids(K * dim);
    for (uint32_t c = 0; c < K; c++) // start from random points, like kmeanspp would
        std::copy(data.begin() + (size_t)(c * 37 % n) * dim, data.begin() + (size_t)(c * 37 % n + 1) * dim,
                  centroids.begin() + (size_t)c * dim);
    diskann::bkmeans::Params p;
    p.K = K;
    p.balanced_iters = 10;
    p.ncand = 4;
    p.block = 512;
    diskann::bkmeans::train_balanced(data.data(), n, dim, p, centroids.data());
    // Every planted blob should now own one centroid: the nearest centroid of each point is
    // shared by all points of its blob, and all K centroids are used.
    std::vector<uint32_t> owner(K, K);
    std::vector<bool> used(K, false);
    for (size_t i = 0; i < n; i++)
    {
        uint32_t best = 0;
        float bd = sqdist(data.data() + i * dim, centroids.data(), dim);
        for (uint32_t c = 1; c < K; c++)
        {
            const float d = sqdist(data.data() + i * dim, centroids.data() + (size_t)c * dim, dim);
            if (d < bd)
            {
                bd = d;
                best = c;
            }
        }
        const uint32_t blob = (uint32_t)(i % K);
        if (owner[blob] == K)
            owner[blob] = best;
        BOOST_TEST(owner[blob] == best);
        used[best] = true;
    }
    for (uint32_t c = 0; c < K; c++)
        BOOST_TEST(used[c]);
}

BOOST_AUTO_TEST_CASE(cluster_base_file_labels_are_nearest)
{
    std::mt19937 rng(5);
    const size_t n = 20000, dim = 16;
    const uint32_t K = 16;
    std::vector<float> data = planted(n, dim, K, rng);
    const std::string path = "bkmeans_test_data.bin";
    diskann::save_bin<float>(path, data.data(), n, dim);

    diskann::bkmeans::Params p;
    p.K = K;
    p.train_points = n; // train on everything
    p.warm_iters = 2;
    p.balanced_iters = 5;
    p.ncand = 4;
    p.block = 4096;
    std::vector<float> centroids;
    std::vector<uint32_t> labels;
    diskann::bkmeans::cluster_base_file<float>(path, p, centroids, labels);
    std::remove(path.c_str());

    BOOST_TEST(centroids.size() == (size_t)K * dim);
    BOOST_TEST(labels.size() == n);
    for (size_t i = 0; i < n; i++)
    {
        uint32_t best = 0;
        float bd = sqdist(data.data() + i * dim, centroids.data(), dim);
        for (uint32_t c = 1; c < K; c++)
        {
            const float d = sqdist(data.data() + i * dim, centroids.data() + (size_t)c * dim, dim);
            if (d < bd)
            {
                bd = d;
                best = c;
            }
        }
        BOOST_TEST(labels[i] == best);
    }
}

BOOST_AUTO_TEST_SUITE_END()
