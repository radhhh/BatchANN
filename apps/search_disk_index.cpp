// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

// Counters of the machine and of this process at one instant. Two of them, taken right before and
// right after the timed query loop, let a driver script compute what happened during the queries
// only; index loading and cache building stay out of the difference.
struct ProcSnapshot
{
    std::string cpu_line;                // aggregate "cpu" line of /proc/stat (whole machine)
    std::vector<std::string> disk_lines; // every line of /proc/diskstats (one per drive or partition)
    // This process only, all threads, from getrusage(). Excludes other programs on the machine.
    uint64_t user_us = 0;        // CPU time in the program's own code
    uint64_t system_us = 0;      // CPU time the kernel spent on its behalf, e.g. submitting reads
    int64_t vol_switches = 0;    // thread went to sleep by itself, e.g. to wait for a read
    int64_t invol_switches = 0;  // thread was taken off its core
};

// Performance record of one L value. Collected in a vector during the search and written to
// --perf_file after the last L, the same way the search results are kept and saved.
struct PerfRecord
{
    uint32_t L = 0;
    uint32_t W = 0;
    uint32_t T = 0;
    ProcSnapshot begin;
    ProcSnapshot end;
    double wall_seconds = 0;
};

ProcSnapshot take_proc_snapshot()
{
    ProcSnapshot snap;
#ifndef _WINDOWS
    std::ifstream stat_file("/proc/stat");
    if (stat_file.is_open())
        std::getline(stat_file, snap.cpu_line);
    std::ifstream disk_file("/proc/diskstats");
    std::string line;
    while (disk_file.is_open() && std::getline(disk_file, line))
        snap.disk_lines.push_back(line);
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
    {
        snap.user_us = (uint64_t)ru.ru_utime.tv_sec * 1000000ULL + (uint64_t)ru.ru_utime.tv_usec;
        snap.system_us = (uint64_t)ru.ru_stime.tv_sec * 1000000ULL + (uint64_t)ru.ru_stime.tv_usec;
        snap.vol_switches = ru.ru_nvcsw;
        snap.invol_switches = ru.ru_nivcsw;
    }
#endif
    return snap;
}

// Text file, one block per L: a PERF header line, the begin and end snapshots as PROCSNAP lines,
// then a WALL line. Parsed by scripts/measure/run_bottleneck.py.
void save_perf_records(const std::string &path, const std::vector<PerfRecord> &records)
{
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    for (const auto &rec : records)
    {
        if (rec.T == 0) // slot of an L value that was skipped (L < K), never filled
            continue;
        out << "PERF L=" << rec.L << " W=" << rec.W << " T=" << rec.T << std::endl;
        const std::pair<const char *, const ProcSnapshot *> snaps[2] = {{"begin", &rec.begin}, {"end", &rec.end}};
        for (const auto &ts : snaps)
        {
            out << "PROCSNAP " << ts.first << " stat " << ts.second->cpu_line << std::endl;
            for (const auto &line : ts.second->disk_lines)
                out << "PROCSNAP " << ts.first << " diskstats " << line << std::endl;
            out << "PROCSNAP " << ts.first << " rusage " << ts.second->user_us << " " << ts.second->system_us << " "
                << ts.second->vol_switches << " " << ts.second->invol_switches << std::endl;
        }
        out << "WALL L=" << rec.L << " seconds=" << rec.wall_seconds << std::endl;
    }
    out.close();
}

// Binary trace file, little-endian. See RESEARCH_PLAN.md / scripts/measure/common.py.
struct TraceHeader
{
    uint32_t magic = 0x54524331; // "TRC1"
    uint32_t version = 1;
    uint32_t nq = 0;
    uint32_t L = 0;
    uint32_t W = 0;
    uint32_t n_cached = 0;
    uint64_t total_records = 0;
    uint32_t reserved[2] = {0, 0};
};

void write_trace(const std::string &path, const diskann::QueryStats *stats, uint64_t nq, uint32_t L, uint32_t W,
                 uint32_t n_cached)
{
    TraceHeader hdr;
    hdr.nq = (uint32_t)nq;
    hdr.L = L;
    hdr.W = W;
    hdr.n_cached = n_cached;
    std::vector<uint32_t> counts(nq), ios(nq), hits(nq), hops(nq);
    uint64_t sum_ios = 0, sum_hits = 0;
    for (uint64_t i = 0; i < nq; i++)
    {
        counts[i] = (uint32_t)(stats[i].trace.size() / 2);
        ios[i] = stats[i].n_ios;
        hits[i] = stats[i].n_cache_hits;
        hops[i] = stats[i].n_hops;
        hdr.total_records += counts[i];
        sum_ios += ios[i];
        sum_hits += hits[i];
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char *>(counts.data()), nq * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(ios.data()), nq * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(hits.data()), nq * sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(hops.data()), nq * sizeof(uint32_t));
    for (uint64_t i = 0; i < nq; i++)
        out.write(reinterpret_cast<const char *>(stats[i].trace.data()), stats[i].trace.size() * sizeof(uint32_t));
    out.close();
    diskann::cout << "TRACE file=" << path << " nq=" << nq << " records=" << hdr.total_records << " sum_ios=" << sum_ios
                  << " sum_hits=" << sum_hits << std::endl;
}

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false,
                      const std::string &trace_file = "", const std::string &cache_list_file = "",
                      const std::string &perf_file = "")
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    // An index built with --num_clusters is renumbered on disk. Its _perm.bin maps a disk id
    // back to the original point id, so results are compared to the original ground truth.
    // Traces and cache lists stay in disk ids on purpose: they describe sectors.
    std::vector<uint32_t> layout_perm;
    {
        std::string perm_file = index_path_prefix + "_disk.index_perm.bin";
        if (file_exists(perm_file))
        {
            std::unique_ptr<uint32_t[]> perm_data;
            size_t perm_n = 0, perm_dim = 0;
            diskann::load_bin<uint32_t>(perm_file, perm_data, perm_n, perm_dim);
            if (perm_dim != 1)
            {
                diskann::cout << "Error. " << perm_file << " must be an N x 1 file" << std::endl;
                return -1;
            }
            layout_perm.assign(perm_data.get(), perm_data.get() + perm_n);
            diskann::cout << "Layout permutation found (" << perm_n
                          << " ids); result ids are mapped back to original ids" << std::endl;
        }
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    // if (num_nodes_to_cache > 0)
    //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
    //     num_threads, node_list);
    if (!cache_list_file.empty())
    {
        diskann::cout << "Cache list holds " << node_list.size() << " nodes" << std::endl;
        diskann::save_bin<uint32_t>(cache_list_file, node_list.data(), node_list.size(), 1);
    }
    const uint32_t n_cached = (uint32_t)node_list.size();
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "Mean IO (us)" << std::setw(16) << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "=================================================================="
                     "================================================================="
                  << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    // One performance record per L, indexed like the results. Filled only when --perf_file is given
    // and saved after the search.
    std::vector<PerfRecord> perf_records(Lvec.size());

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];
        if (!trace_file.empty())
            for (size_t i = 0; i < query_num; i++)
                stats[i].record_trace = true;

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        if (!perf_file.empty())
            perf_records[test_id].begin = take_proc_snapshot();
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (!filtered_search)
            {
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());
        if (!perf_file.empty())
        {
            perf_records[test_id].end = take_proc_snapshot();
            perf_records[test_id].L = L;
            perf_records[test_id].W = optimized_beamwidth;
            perf_records[test_id].T = num_threads;
            perf_records[test_id].wall_seconds = diff.count();
        }

        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);
        if (!layout_perm.empty())
        {
            for (auto &id : query_result_ids[test_id])
            {
                if (id >= layout_perm.size())
                    throw diskann::ANNException("Result id outside the layout permutation; the _perm.bin file "
                                                "does not belong to this index",
                                                -1, __FUNCSIG__, __FILE__, __LINE__);
                id = layout_perm[id];
            }
        }

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.io_us; });

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;
        if (!trace_file.empty())
            write_trace(trace_file + "_L" + std::to_string(L) + ".trace", stats, query_num, L, optimized_beamwidth,
                        n_cached);
        delete[] stats;
    }

    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }
    if (!perf_file.empty())
        save_perf_records(perf_file, perf_records);

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
        label_type, query_filters_file, trace_file, cache_list_file, perf_file;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("trace_file", po::value<std::string>(&trace_file)->default_value(""),
                                       "If set, record every expanded node per query (hop, cache/disk) and write "
                                       "<trace_file>_L<L>.trace for each L. Default: off");
        optional_configs.add_options()("perf_file", po::value<std::string>(&perf_file)->default_value(""),
                                       "If set, write a performance record to this text file: machine CPU, drive and "
                                       "process counters right before and after the timed query loop, plus the wall "
                                       "time, one block per L. Used by scripts/measure/run_bottleneck.py. Default: off");
        optional_configs.add_options()("cache_list_file", po::value<std::string>(&cache_list_file)->default_value(""),
                                       "If set, save the ids of the statically cached nodes to this .bin file");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, trace_file, cache_list_file, perf_file);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, trace_file, cache_list_file, perf_file);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data, trace_file, cache_list_file, perf_file);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data, trace_file, cache_list_file, perf_file);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data, trace_file, cache_list_file, perf_file);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data, trace_file, cache_list_file, perf_file);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}