// See include/layout_reorder.h.
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <deque>
#include <fstream>
#include <limits>
#include <numeric>

#include <omp.h>

#include "layout_reorder.h"
#include "ann_exception.h"
#include "balanced_kmeans.h"
#include "cached_io.h"
#include "defaults.h"
#include "logger.h"
#include "timer.h"
#include "utils.h"

namespace diskann
{
namespace layout
{

static constexpr uint32_t NONE = std::numeric_limits<uint32_t>::max();
static constexpr uint64_t IO_BUF = 64 * 1024 * 1024;

LayoutMode parse_layout_mode(const std::string &s)
{
    if (s == "cluster_only")
        return LayoutMode::cluster_only;
    if (s == "cluster_neighbor")
        return LayoutMode::cluster_neighbor;
    throw diskann::ANNException("Unknown --layout_mode '" + s + "' (use cluster_only or cluster_neighbor)", -1,
                                __FUNCSIG__, __FILE__, __LINE__);
}

Graph load_mem_index_graph(const std::string &path)
{
    const size_t file_size = get_file_size(path);
    cached_ifstream in(path, IO_BUF);
    uint64_t expected_size, num_frozen;
    Graph g;
    in.read((char *)&expected_size, sizeof(uint64_t));
    in.read((char *)&g.max_degree, sizeof(uint32_t));
    in.read((char *)&g.start, sizeof(uint32_t));
    in.read((char *)&num_frozen, sizeof(uint64_t));
    if (expected_size != file_size)
        throw diskann::ANNException("Graph file size does not match its header: " + path, -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    if (num_frozen != 0)
        throw diskann::ANNException("Layout reorder does not support frozen points", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);

    const size_t words = (file_size - 24) / sizeof(uint32_t);
    g.nbrs.reserve(words);
    g.offsets.reserve(words / std::max<size_t>(1, g.max_degree) + 1);
    g.offsets.push_back(0);
    size_t bytes_read = 24;
    while (bytes_read < file_size)
    {
        uint32_t k;
        in.read((char *)&k, sizeof(uint32_t));
        const size_t old = g.nbrs.size();
        g.nbrs.resize(old + k);
        in.read((char *)(g.nbrs.data() + old), (uint64_t)k * sizeof(uint32_t));
        g.offsets.push_back(g.nbrs.size());
        bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
    }
    if (bytes_read != file_size)
        throw diskann::ANNException("Graph file ended mid-record: " + path, -1, __FUNCSIG__, __FILE__, __LINE__);
    const size_t N = g.n();
    for (uint32_t u : g.nbrs)
        if (u >= N)
            throw diskann::ANNException("Graph file has a neighbor id out of range: " + path, -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
    diskann::cout << "Loaded graph: " << N << " nodes, " << g.nbrs.size() << " edges, max degree " << g.max_degree
                  << ", start " << g.start << std::endl;
    return g;
}

std::vector<uint32_t> invert(const std::vector<uint32_t> &perm)
{
    std::vector<uint32_t> inv(perm.size(), NONE);
    for (size_t i = 0; i < perm.size(); i++)
    {
        if (perm[i] >= perm.size() || inv[perm[i]] != NONE)
            throw diskann::ANNException("perm is not a permutation", -1, __FUNCSIG__, __FILE__, __LINE__);
        inv[perm[i]] = (uint32_t)i;
    }
    return inv;
}

std::vector<uint32_t> compute_layout_order(const Graph &g, const std::vector<uint32_t> &labels, uint32_t K,
                                           uint32_t S, LayoutMode mode, std::vector<uint32_t> &cluster_ranges)
{
    const size_t N = g.n();
    if (labels.size() != N)
        throw diskann::ANNException("labels.size() != number of graph nodes", -1, __FUNCSIG__, __FILE__, __LINE__);
    if (S == 0 || K == 0)
        throw diskann::ANNException("compute_layout_order: S and K must be positive", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);

    // 1. group by cluster: members = old ids ordered by (label, id)
    std::vector<uint32_t> count(K, 0);
    for (uint32_t l : labels)
    {
        if (l >= K)
            throw diskann::ANNException("label out of range", -1, __FUNCSIG__, __FILE__, __LINE__);
        count[l]++;
    }
    std::vector<uint32_t> first(K + 1, 0);
    for (uint32_t c = 0; c < K; c++)
        first[c + 1] = first[c] + count[c];
    std::vector<uint32_t> members(N), fill(K, 0);
    for (size_t v = 0; v < N; v++)
        members[first[labels[v]] + fill[labels[v]]++] = (uint32_t)v;
    cluster_ranges.resize((size_t)K * 2);
    for (uint32_t c = 0; c < K; c++)
    {
        cluster_ranges[(size_t)c * 2] = first[c];
        cluster_ranges[(size_t)c * 2 + 1] = first[c + 1];
    }
    if (mode == LayoutMode::cluster_only)
        return members;

    // 2. greedy sector filling inside each cluster
    std::vector<uint32_t> perm(N);
    std::vector<uint8_t> placed(N, 0);
    std::vector<uint32_t> score(N, 0); // edges from the nodes already in the current sector; 0 = not a candidate
    std::vector<uint32_t> touched;     // ids with a non-zero score (plus some already placed, skipped on scan)
    std::deque<uint32_t> pending;      // candidates seen from earlier sectors of this cluster
    size_t pos = 0;
    for (uint32_t c = 0; c < K; c++)
    {
        size_t seed = first[c];
        pending.clear();
        touched.clear();
        for (uint32_t step = 0; step < count[c]; step++)
        {
            uint32_t v = NONE, best = 0;
            for (uint32_t u : touched)
                if (score[u] > best || (score[u] == best && best > 0 && u < v))
                {
                    best = score[u];
                    v = u;
                }
            if (best == 0)
            {
                v = NONE;
                while (!pending.empty())
                {
                    const uint32_t u = pending.front();
                    pending.pop_front();
                    if (!placed[u])
                    {
                        v = u;
                        break;
                    }
                }
                if (v == NONE)
                {
                    while (placed[members[seed]])
                        seed++;
                    v = members[seed];
                }
            }
            perm[pos++] = v;
            placed[v] = 1;
            score[v] = 0;
            const uint32_t *lst = g.list(v);
            const uint32_t deg = g.degree(v);
            for (uint32_t i = 0; i < deg; i++)
            {
                const uint32_t u = lst[i];
                if (!placed[u] && labels[u] == c)
                {
                    if (score[u] == 0)
                        touched.push_back(u);
                    score[u]++;
                }
            }
            if (pos % S == 0)
            {
                std::sort(touched.begin(), touched.end(), [&score](uint32_t a, uint32_t b) {
                    return score[a] > score[b] || (score[a] == score[b] && a < b);
                });
                for (uint32_t u : touched)
                {
                    if (score[u] > 0)
                        pending.push_back(u);
                    score[u] = 0;
                }
                touched.clear();
            }
        }
        for (uint32_t u : touched)
            score[u] = 0;
    }
    return perm;
}

LayoutStats layout_stats(const Graph &g, const std::vector<uint32_t> &inv, const std::vector<uint32_t> &labels,
                         uint32_t K, uint32_t S)
{
    const size_t N = g.n();
    LayoutStats st;
    uint64_t same_sector = 0, same_cluster = 0;
#pragma omp parallel for schedule(static) reduction(+ : same_sector, same_cluster)
    for (int64_t v = 0; v < (int64_t)N; v++)
    {
        const uint64_t sv = inv[v] / S;
        const uint32_t *lst = g.list(v);
        const uint32_t deg = g.degree(v);
        for (uint32_t i = 0; i < deg; i++)
        {
            same_sector += (inv[lst[i]] / S == sv);
            same_cluster += (labels[lst[i]] == labels[v]);
        }
    }
    const double edges = (double)std::max<size_t>(1, g.nbrs.size());
    st.same_sector_edges = (double)same_sector / edges;
    st.same_cluster_edges = (double)same_cluster / edges;

    std::vector<uint64_t> key(N);
    std::vector<uint32_t> count(K, 0);
    for (size_t v = 0; v < N; v++)
    {
        key[v] = ((uint64_t)labels[v] << 32) | (inv[v] / S);
        count[labels[v]]++;
    }
    std::sort(key.begin(), key.end());
    std::vector<uint64_t> sectors(K, 0);
    for (size_t i = 0; i < N; i++)
        if (i == 0 || key[i] != key[i - 1])
            sectors[key[i] >> 32]++;
    double sum = 0, ideal = 0;
    for (uint32_t c = 0; c < K; c++)
    {
        sum += (double)sectors[c];
        ideal += (double)((count[c] + S - 1) / S);
        st.max_sectors_per_cluster = std::max(st.max_sectors_per_cluster, sectors[c]);
    }
    st.mean_sectors_per_cluster = sum / (double)K;
    st.ideal_mean_sectors_per_cluster = ideal / (double)K;
    return st;
}

void write_permuted_graph(const Graph &g, const std::vector<uint32_t> &perm, const std::vector<uint32_t> &inv,
                          const std::string &out_path)
{
    const size_t N = g.n();
    if (perm.size() != N || inv.size() != N)
        throw diskann::ANNException("write_permuted_graph: perm size != graph size", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    uint64_t size = 24;
    uint32_t max_degree = g.max_degree;
    uint32_t start = inv[g.start];
    uint64_t frozen = 0;
    {
        cached_ofstream out(out_path, IO_BUF);
        out.write((char *)&size, sizeof(uint64_t));
        out.write((char *)&max_degree, sizeof(uint32_t));
        out.write((char *)&start, sizeof(uint32_t));
        out.write((char *)&frozen, sizeof(uint64_t));
        uint32_t maxk = 0;
        for (size_t v = 0; v < N; v++)
            maxk = std::max(maxk, g.degree(v));
        std::vector<uint32_t> buf((size_t)maxk + 1);
        for (size_t nw = 0; nw < N; nw++)
        {
            const uint32_t old = perm[nw];
            const uint32_t k = g.degree(old);
            const uint32_t *lst = g.list(old);
            buf[0] = k;
            for (uint32_t i = 0; i < k; i++)
                buf[1 + i] = inv[lst[i]];
            out.write((char *)buf.data(), ((uint64_t)k + 1) * sizeof(uint32_t));
            size += ((uint64_t)k + 1) * sizeof(uint32_t);
        }
        out.close();
    }
    std::fstream f(out_path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(0, std::ios::beg);
    f.write((char *)&size, sizeof(uint64_t));
    f.close();
}

template <typename T>
void write_permuted_rows(const std::string &in_path, const std::string &out_path, const std::vector<uint32_t> &perm)
{
    std::unique_ptr<T[]> data;
    size_t npts, dim;
    diskann::load_bin<T>(in_path, data, npts, dim);
    if (npts != perm.size())
        throw diskann::ANNException("write_permuted_rows: row count of " + in_path + " != perm size", -1, __FUNCSIG__,
                                    __FILE__, __LINE__);
    cached_ofstream out(out_path, IO_BUF);
    uint32_t npts32 = (uint32_t)npts, dim32 = (uint32_t)dim;
    out.write((char *)&npts32, sizeof(uint32_t));
    out.write((char *)&dim32, sizeof(uint32_t));
    const size_t block = std::min<size_t>(npts, 1u << 20);
    std::vector<T> buf(block * dim);
    for (size_t s = 0; s < npts; s += block)
    {
        const size_t nb = std::min(block, npts - s);
        for (size_t i = 0; i < nb; i++)
            std::memcpy(buf.data() + i * dim, data.get() + (size_t)perm[s + i] * dim, dim * sizeof(T));
        out.write((char *)buf.data(), nb * dim * sizeof(T));
    }
    out.close();
}

void write_remapped_ids(const std::string &in_path, const std::string &out_path, const std::vector<uint32_t> &inv)
{
    std::unique_ptr<uint32_t[]> ids;
    size_t m, dim;
    diskann::load_bin<uint32_t>(in_path, ids, m, dim);
    if (dim != 1)
        throw diskann::ANNException("write_remapped_ids: expected an m x 1 file: " + in_path, -1, __FUNCSIG__,
                                    __FILE__, __LINE__);
    for (size_t i = 0; i < m; i++)
    {
        if (ids[i] >= inv.size())
            throw diskann::ANNException("write_remapped_ids: id out of range in " + in_path, -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
        ids[i] = inv[ids[i]];
    }
    diskann::save_bin<uint32_t>(out_path, ids.get(), m, 1);
}

template DISKANN_DLLEXPORT void write_permuted_rows<float>(const std::string &, const std::string &,
                                                           const std::vector<uint32_t> &);
template DISKANN_DLLEXPORT void write_permuted_rows<int8_t>(const std::string &, const std::string &,
                                                            const std::vector<uint32_t> &);
template DISKANN_DLLEXPORT void write_permuted_rows<uint8_t>(const std::string &, const std::string &,
                                                             const std::vector<uint32_t> &);

static void rename_over(const std::string &tmp, const std::string &dst)
{
    if (std::rename(tmp.c_str(), dst.c_str()) != 0)
        throw diskann::ANNException("Could not rename " + tmp + " over " + dst, -1, __FUNCSIG__, __FILE__, __LINE__);
}

static void write_stats(std::ofstream &f, const char *tag, const LayoutStats &s)
{
    f << tag << ".same_sector_edges=" << s.same_sector_edges << "\n"
      << tag << ".same_cluster_edges=" << s.same_cluster_edges << "\n"
      << tag << ".mean_sectors_per_cluster=" << s.mean_sectors_per_cluster << "\n"
      << tag << ".ideal_mean_sectors_per_cluster=" << s.ideal_mean_sectors_per_cluster << "\n"
      << tag << ".max_sectors_per_cluster=" << s.max_sectors_per_cluster << "\n";
}

} // namespace layout

template <typename T>
void reorder_for_layout(const std::string &data_file, const std::string &mem_index_path,
                        const std::string &pq_compressed_path, const std::string &disk_pq_path,
                        const std::string &medoids_path, const std::string &layout_data_out,
                        const std::string &extra_prefix, size_t payload_bytes, uint32_t num_clusters,
                        const std::string &layout_mode, uint32_t num_threads)
{
    using namespace diskann::layout;
    (void)num_threads; // the build has already set the OpenMP and MKL thread counts
    const LayoutMode mode = parse_layout_mode(layout_mode);
    Timer timer;

    // 1. cluster labels, one per old id
    bkmeans::Params params;
    params.K = num_clusters;
    std::vector<float> centroids;
    std::vector<uint32_t> labels;
    bkmeans::cluster_base_file<T>(data_file, params, centroids, labels);
    const size_t N = labels.size();

    // 2. graph and nodes per sector, computed exactly as create_disk_layout will
    Graph g = load_mem_index_graph(mem_index_path);
    if (g.n() != N)
        throw diskann::ANNException("Graph node count != data point count", -1, __FUNCSIG__, __FILE__, __LINE__);
    const uint64_t max_node_len = payload_bytes + ((uint64_t)g.max_degree + 1) * sizeof(uint32_t);
    const uint32_t S = (uint32_t)(defaults::SECTOR_LEN / max_node_len);
    if (S == 0)
        throw diskann::ANNException("Layout reorder needs at least one node per sector (node is " +
                                        std::to_string(max_node_len) + " bytes)",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    diskann::cout << "Layout reorder: " << S << " nodes per sector, mode " << layout_mode << std::endl;

    // 3. the new numbering
    timer.reset();
    std::vector<uint32_t> cluster_ranges;
    std::vector<uint32_t> perm = compute_layout_order(g, labels, num_clusters, S, mode, cluster_ranges);
    std::vector<uint32_t> inv = invert(perm);
    diskann::cout << timer.elapsed_seconds_for_step("choosing the layout order") << std::endl;

    // 4. sector statistics of the old and the new order
    LayoutStats st_old, st_new;
    {
        std::vector<uint32_t> identity(N);
        std::iota(identity.begin(), identity.end(), 0u);
        st_old = layout_stats(g, identity, labels, num_clusters, S);
    }
    st_new = layout_stats(g, inv, labels, num_clusters, S);
    diskann::cout << "Same-sector edges: old " << st_old.same_sector_edges << " -> new " << st_new.same_sector_edges
                  << "; same-cluster edges " << st_new.same_cluster_edges << "; sectors per cluster: old "
                  << st_old.mean_sectors_per_cluster << " -> new " << st_new.mean_sectors_per_cluster << " (ideal "
                  << st_new.ideal_mean_sectors_per_cluster << ", max " << st_new.max_sectors_per_cluster << ")"
                  << std::endl;

    // 5. the files, in the new numbering
    timer.reset();
    write_permuted_graph(g, perm, inv, mem_index_path + ".tmp");
    rename_over(mem_index_path + ".tmp", mem_index_path);
    g = Graph();
    write_permuted_rows<T>(data_file, layout_data_out, perm);
    write_permuted_rows<uint8_t>(pq_compressed_path, pq_compressed_path + ".tmp", perm);
    rename_over(pq_compressed_path + ".tmp", pq_compressed_path);
    if (!disk_pq_path.empty() && file_exists(disk_pq_path))
    {
        write_permuted_rows<uint8_t>(disk_pq_path, disk_pq_path + ".tmp", perm);
        rename_over(disk_pq_path + ".tmp", disk_pq_path);
    }
    if (file_exists(medoids_path))
    {
        write_remapped_ids(medoids_path, medoids_path + ".tmp", inv);
        rename_over(medoids_path + ".tmp", medoids_path);
    }

    // 6. extra files next to the index
    diskann::save_bin<uint32_t>(extra_prefix + "_perm.bin", perm.data(), N, 1);
    diskann::save_bin<uint32_t>(extra_prefix + "_inv.bin", inv.data(), N, 1);
    diskann::save_bin<uint32_t>(extra_prefix + "_cluster_ranges.bin", cluster_ranges.data(), num_clusters, 2);
    diskann::save_bin<float>(extra_prefix + "_kmeans_centroids.bin", centroids.data(), num_clusters,
                             centroids.size() / num_clusters);
    {
        std::vector<uint32_t> labels_new(N);
        for (size_t nw = 0; nw < N; nw++)
            labels_new[nw] = labels[perm[nw]];
        diskann::save_bin<uint32_t>(extra_prefix + "_labels.bin", labels_new.data(), N, 1);
    }
    {
        std::ofstream f(extra_prefix + "_layout_stats.txt");
        f << "S=" << S << "\nK=" << num_clusters << "\nmode=" << layout_mode << "\nN=" << N << "\n";
        write_stats(f, "old", st_old);
        write_stats(f, "new", st_new);
    }
    diskann::cout << timer.elapsed_seconds_for_step("writing the reordered files") << std::endl;
}

template DISKANN_DLLEXPORT void reorder_for_layout<float>(const std::string &, const std::string &,
                                                          const std::string &, const std::string &,
                                                          const std::string &, const std::string &,
                                                          const std::string &, size_t, uint32_t, const std::string &,
                                                          uint32_t);
template DISKANN_DLLEXPORT void reorder_for_layout<int8_t>(const std::string &, const std::string &,
                                                           const std::string &, const std::string &,
                                                           const std::string &, const std::string &,
                                                           const std::string &, size_t, uint32_t, const std::string &,
                                                           uint32_t);
template DISKANN_DLLEXPORT void reorder_for_layout<uint8_t>(const std::string &, const std::string &,
                                                            const std::string &, const std::string &,
                                                            const std::string &, const std::string &,
                                                            const std::string &, size_t, uint32_t,
                                                            const std::string &, uint32_t);

} // namespace diskann
