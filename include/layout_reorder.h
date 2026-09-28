// Locality-aware disk layout: renumber the points of a built Vamana graph so that points of
// one cluster are contiguous and, inside a cluster, graph neighbors share a sector
// (RESEARCH_PLAN.md, contribution 4). The disk position of node i depends only on i, so
// renumbering is enough; the reader is untouched.
//
// Vocabulary: perm[new] = old (which old point sits at new id), inv[old] = new.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "windows_customizations.h"

namespace diskann
{
namespace layout
{

// Adjacency lists of a _mem.index file in two flat arrays.
struct Graph
{
    uint32_t max_degree = 0;
    uint32_t start = 0;              // entry point (medoid)
    std::vector<uint64_t> offsets;   // N + 1; the list of node v is nbrs[offsets[v] .. offsets[v+1])
    std::vector<uint32_t> nbrs;
    size_t n() const
    {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }
    uint32_t degree(size_t v) const
    {
        return (uint32_t)(offsets[v + 1] - offsets[v]);
    }
    const uint32_t *list(size_t v) const
    {
        return nbrs.data() + offsets[v];
    }
};

enum class LayoutMode
{
    cluster_only,     // clusters back to back, old relative order inside a cluster
    cluster_neighbor  // plus greedy sector filling by graph adjacency inside a cluster
};

DISKANN_DLLEXPORT LayoutMode parse_layout_mode(const std::string &s); // throws on an unknown value

// Streams a _mem.index file (header + per-node lists). Refuses frozen points.
DISKANN_DLLEXPORT Graph load_mem_index_graph(const std::string &path);

// labels: cluster of every old id, values in [0, K). S: nodes per sector.
// Returns perm. cluster_ranges receives K pairs [first, one_past_last) of new ids.
DISKANN_DLLEXPORT std::vector<uint32_t> compute_layout_order(const Graph &g, const std::vector<uint32_t> &labels,
                                                             uint32_t K, uint32_t S, LayoutMode mode,
                                                             std::vector<uint32_t> &cluster_ranges);

DISKANN_DLLEXPORT std::vector<uint32_t> invert(const std::vector<uint32_t> &perm);

struct LayoutStats
{
    double same_sector_edges = 0;        // fraction of edges (v,u) with inv[v]/S == inv[u]/S
    double same_cluster_edges = 0;       // fraction with labels[v] == labels[u] (independent of the order)
    double mean_sectors_per_cluster = 0;
    double ideal_mean_sectors_per_cluster = 0; // mean over clusters of ceil(size / S)
    uint64_t max_sectors_per_cluster = 0;
};

// Statistics of an order given by inv (pass the identity for the old order).
DISKANN_DLLEXPORT LayoutStats layout_stats(const Graph &g, const std::vector<uint32_t> &inv,
                                           const std::vector<uint32_t> &labels, uint32_t K, uint32_t S);

// Graph file in new-id order, neighbor ids and the medoid replaced by their new ids.
DISKANN_DLLEXPORT void write_permuted_graph(const Graph &g, const std::vector<uint32_t> &perm,
                                            const std::vector<uint32_t> &inv, const std::string &out_path);

// Any .bin file with fixed-size rows of T (vectors, PQ codes): rows in new-id order.
template <typename T>
DISKANN_DLLEXPORT void write_permuted_rows(const std::string &in_path, const std::string &out_path,
                                           const std::vector<uint32_t> &perm);

// A .bin of uint32 ids (m x 1): every id replaced by inv[id].
DISKANN_DLLEXPORT void write_remapped_ids(const std::string &in_path, const std::string &out_path,
                                          const std::vector<uint32_t> &inv);

} // namespace layout

// Step 3b of build_disk_index: cluster, choose the numbering, write the files again.
//   data_file            vectors to read (base, or the prepped copy for mips/cosine); never modified
//   mem_index_path       P_mem.index, replaced by the renumbered graph
//   pq_compressed_path   P_pq_compressed.bin, rows reordered in place
//   disk_pq_path         P_disk.index_pq_compressed.bin, same, or "" when disk PQ is off
//   medoids_path         P_disk.index_medoids.bin, ids remapped if the file exists
//   layout_data_out      where the reordered vectors are written
//   extra_prefix         prefix of the extra files (P_disk.index -> P_disk.index_perm.bin, ...)
//   payload_bytes        bytes stored per node before the neighbor list (dim*sizeof(T), or disk PQ bytes)
template <typename T>
DISKANN_DLLEXPORT void reorder_for_layout(const std::string &data_file, const std::string &mem_index_path,
                                          const std::string &pq_compressed_path, const std::string &disk_pq_path,
                                          const std::string &medoids_path, const std::string &layout_data_out,
                                          const std::string &extra_prefix, size_t payload_bytes, uint32_t num_clusters,
                                          const std::string &layout_mode, uint32_t num_threads);

} // namespace diskann
