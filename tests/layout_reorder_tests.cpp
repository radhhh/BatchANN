// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include <cstdio>
#include <fstream>
#include <set>
#include <vector>

#include "layout_reorder.h"
#include "utils.h"

namespace
{
// Writes lists in the _mem.index format (see in_mem_graph_store.cpp save_graph).
void write_graph_file(const std::string &path, const std::vector<std::vector<uint32_t>> &lists, uint32_t start)
{
    std::ofstream out(path, std::ios::binary);
    uint64_t size = 24, frozen = 0;
    uint32_t max_degree = 0;
    for (auto &l : lists)
        max_degree = std::max(max_degree, (uint32_t)l.size());
    out.write((char *)&size, 8);
    out.write((char *)&max_degree, 4);
    out.write((char *)&start, 4);
    out.write((char *)&frozen, 8);
    for (auto &l : lists)
    {
        uint32_t k = (uint32_t)l.size();
        out.write((char *)&k, 4);
        out.write((char *)l.data(), 4 * k);
        size += 4 * ((uint64_t)k + 1);
    }
    out.seekp(0);
    out.write((char *)&size, 8);
}

// Two clusters of six nodes, {0..5} and {6..11}, each with the same edge pattern, plus two
// cross-cluster edges. With S = 3 the greedy rule fills the first sector of cluster 0 with
// 0, 3, 5 (0 -> 3, 3 -> 5) and the second with 1, 2, 4; same shape in cluster 1.
const std::vector<std::vector<uint32_t>> LISTS = {
    {3, 5, 7}, {2, 4}, {1}, {0, 5}, {1}, {0, 3}, {9, 11}, {8, 10}, {7}, {6, 11}, {7}, {6, 9, 0},
};
const std::vector<uint32_t> LABELS = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
const std::vector<uint32_t> EXPECTED_PERM = {0, 3, 5, 1, 2, 4, 6, 9, 11, 7, 8, 10};
} // namespace

BOOST_AUTO_TEST_SUITE(layout_reorder_tests)

BOOST_AUTO_TEST_CASE(order_and_graph_rewrite)
{
    using namespace diskann::layout;
    const std::string path = "layout_test_graph.bin", path2 = "layout_test_graph2.bin";
    write_graph_file(path, LISTS, /*start=*/5);

    Graph g = load_mem_index_graph(path);
    BOOST_TEST(g.n() == 12u);
    BOOST_TEST(g.start == 5u);
    BOOST_TEST(g.max_degree == 3u);
    BOOST_TEST(g.degree(11) == 3u);
    BOOST_TEST(g.list(1)[1] == 4u);

    std::vector<uint32_t> ranges;
    std::vector<uint32_t> perm = compute_layout_order(g, LABELS, 2, 3, LayoutMode::cluster_neighbor, ranges);
    BOOST_TEST(perm == EXPECTED_PERM, boost::test_tools::per_element());
    BOOST_TEST(ranges == (std::vector<uint32_t>{0, 6, 6, 12}), boost::test_tools::per_element());

    std::vector<uint32_t> co = compute_layout_order(g, LABELS, 2, 3, LayoutMode::cluster_only, ranges);
    for (uint32_t i = 0; i < 12; i++)
        BOOST_TEST(co[i] == i);

    std::vector<uint32_t> inv = invert(perm);
    for (uint32_t i = 0; i < 12; i++)
        BOOST_TEST(inv[perm[i]] == i);

    std::vector<uint32_t> identity(12);
    for (uint32_t i = 0; i < 12; i++)
        identity[i] = i;
    LayoutStats old_st = layout_stats(g, identity, LABELS, 2, 3);
    LayoutStats new_st = layout_stats(g, inv, LABELS, 2, 3);
    BOOST_TEST(std::abs(old_st.same_sector_edges - 8.0 / 22.0) < 1e-9);
    BOOST_TEST(std::abs(new_st.same_sector_edges - 20.0 / 22.0) < 1e-9);
    BOOST_TEST(std::abs(new_st.same_cluster_edges - 20.0 / 22.0) < 1e-9);
    BOOST_TEST(new_st.mean_sectors_per_cluster == 2.0);
    BOOST_TEST(new_st.ideal_mean_sectors_per_cluster == 2.0);
    BOOST_TEST(new_st.max_sectors_per_cluster == 2u);

    write_permuted_graph(g, perm, inv, path2);
    Graph g2 = load_mem_index_graph(path2);
    BOOST_TEST(g2.n() == 12u);
    BOOST_TEST(g2.max_degree == g.max_degree);
    BOOST_TEST(g2.start == inv[g.start]);
    for (uint32_t nw = 0; nw < 12; nw++)
    {
        std::set<uint32_t> expect, got;
        const uint32_t old = perm[nw];
        for (uint32_t i = 0; i < g.degree(old); i++)
            expect.insert(inv[g.list(old)[i]]);
        for (uint32_t i = 0; i < g2.degree(nw); i++)
            got.insert(g2.list(nw)[i]);
        BOOST_TEST(expect == got);
    }
    std::remove(path.c_str());
    std::remove(path2.c_str());
}

BOOST_AUTO_TEST_CASE(rows_and_ids)
{
    using namespace diskann::layout;
    const std::string in = "layout_test_rows.bin", out = "layout_test_rows2.bin";
    const std::string ids_in = "layout_test_ids.bin", ids_out = "layout_test_ids2.bin";
    std::vector<uint8_t> rows(5 * 3);
    for (size_t i = 0; i < rows.size(); i++)
        rows[i] = (uint8_t)(10 * (i / 3) + i % 3);
    diskann::save_bin<uint8_t>(in, rows.data(), 5, 3);
    std::vector<uint32_t> perm = {4, 2, 0, 1, 3};
    write_permuted_rows<uint8_t>(in, out, perm);
    std::unique_ptr<uint8_t[]> got;
    size_t n, d;
    diskann::load_bin<uint8_t>(out, got, n, d);
    BOOST_TEST(n == 5u);
    BOOST_TEST(d == 3u);
    for (size_t nw = 0; nw < 5; nw++)
        for (size_t k = 0; k < 3; k++)
            BOOST_TEST(got[nw * 3 + k] == rows[perm[nw] * 3 + k]);

    std::vector<uint32_t> inv = invert(perm);
    std::vector<uint32_t> ids = {0, 4};
    diskann::save_bin<uint32_t>(ids_in, ids.data(), 2, 1);
    write_remapped_ids(ids_in, ids_out, inv);
    std::unique_ptr<uint32_t[]> got_ids;
    diskann::load_bin<uint32_t>(ids_out, got_ids, n, d);
    BOOST_TEST(n == 2u);
    BOOST_TEST(got_ids[0] == inv[0]);
    BOOST_TEST(got_ids[1] == inv[4]);
    std::remove(in.c_str());
    std::remove(out.c_str());
    std::remove(ids_in.c_str());
    std::remove(ids_out.c_str());
}

BOOST_AUTO_TEST_SUITE_END()
