// metrics.cpp — a COMMON quality evaluator for the SparseBinSOM-vs-somoclu comparison.
//
// Scores any trained SOM codebook against a sparse-binary corpus with identical code, so
// the two libraries are judged on the same footing. Computes, under COSINE distance:
//   * QE  — mean cosine distance (1 - cos) from each sample to its best matching unit (BMU)
//   * TE  — topographic error: fraction of samples whose 1st and 2nd BMUs are NOT adjacent
//           on the map grid (Chebyshev distance > 1, i.e. not in the 8-neighbourhood)
//
// Reads two codebook formats into one feature-major layout W[v*K + k] (feature v, neuron k):
//   * somw : SparseBinSOM .somw — 24-byte header then K*V float32, already feature-major
//   * wts  : somoclu .wts — text; "%Y X", "%V", then K lines of V floats (neuron-major)
//
// Sample is binary (presence), so ||sample|| = sqrt(nnz) and the dot with neuron k is the
// sum of that neuron's weights over the sample's present features. C++/OpenMP, no deps —
// pure-Python can't do cosine over millions of samples × thousands of neurons in time.
//
// Build:  g++ -O3 -fopenmp tools/metrics.cpp -o tools/metrics
// Usage:  tools/metrics --corpus c.sbcsr --codebook m.wts --format wts [--rows R --cols C]
//                       [--sample N] [--seed S]   (prints a JSON line of metrics)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Corpus {           // CSR sparse-binary: row_ptr[n+1], col_idx[nnz]
    uint32_t n_samples = 0, n_features = 0, n_nonzeros = 0;
    std::vector<uint32_t> row_ptr;
    std::vector<uint16_t> col_idx;
};

static Corpus load_sbcsr(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open corpus " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "SCSR1\0\0\0", 8) != 0 &&            // .scsr
        std::memcmp(magic, "SBCSR1\0\0", 8) != 0)               // .sbcsr (legacy)
        throw std::runtime_error("not a .scsr/.sbcsr: " + path);
    uint32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), 16);
    Corpus c;
    c.n_samples = hdr[0]; c.n_features = hdr[1]; c.n_nonzeros = hdr[2];
    c.row_ptr.resize(c.n_samples + 1);
    c.col_idx.resize(c.n_nonzeros);
    f.read(reinterpret_cast<char*>(c.row_ptr.data()), (std::streamsize)(c.n_samples + 1) * 4);
    f.read(reinterpret_cast<char*>(c.col_idx.data()), (std::streamsize)c.n_nonzeros * 2);
    if (!f) throw std::runtime_error("truncated corpus " + path);
    return c;
}

// Load a codebook into feature-major W[v*K + k]; sets K, V, and grid dims (rows, cols).
static std::vector<float> load_codebook(const std::string& path, const std::string& fmt,
                                        uint32_t& K, uint32_t& V,
                                        uint32_t& rows, uint32_t& cols,
                                        uint32_t cli_rows, uint32_t cli_cols) {
    std::vector<float> W;
    if (fmt == "somw") {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open codebook " + path);
        char magic[8];
        f.read(magic, 8);
        if (std::memcmp(magic, "SSOMW1\0\0", 8) != 0 &&         // .ssomw
            std::memcmp(magic, "SBSOMW1\0", 8) != 0)            // .somw (legacy)
            throw std::runtime_error("not a .ssomw/.somw: " + path);
        uint32_t h[4];
        f.read(reinterpret_cast<char*>(h), 16);     // map_rows, map_cols, n_features, reserved
        rows = h[0]; cols = h[1]; V = h[2]; K = rows * cols;
        W.resize((size_t)V * K);                      // already feature-major W[v*K + k]
        f.read(reinterpret_cast<char*>(W.data()), (std::streamsize)W.size() * sizeof(float));
        if (!f) throw std::runtime_error("truncated .somw " + path);
    } else if (fmt == "wts") {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open codebook " + path);
        std::string line;
        // Header: "%<Y> <X>" then "%<V>". Y=rows, X=cols.
        std::getline(f, line); std::sscanf(line.c_str(), "%%%u %u", &rows, &cols);
        std::getline(f, line); std::sscanf(line.c_str(), "%%%u", &V);
        K = rows * cols;
        if (K == 0 && cli_rows && cli_cols) { rows = cli_rows; cols = cli_cols; K = rows * cols; }
        W.assign((size_t)V * K, 0.0f);
        for (uint32_t k = 0; k < K; ++k)              // neuron-major text -> feature-major store
            for (uint32_t v = 0; v < V; ++v) {
                float w;
                if (!(f >> w)) throw std::runtime_error("truncated .wts at neuron " + std::to_string(k));
                W[(size_t)v * K + k] = w;
            }
    } else {
        throw std::runtime_error("unknown --format " + fmt + " (use somw|wts)");
    }
    return W;
}

int main(int argc, char** argv) {
    std::string corpus_path, cb_path, fmt = "wts";
    uint32_t sample = 0, seed = 0, cli_rows = 0, cli_cols = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if      (a == "--corpus")   corpus_path = nxt();
        else if (a == "--codebook") cb_path = nxt();
        else if (a == "--format")   fmt = nxt();
        else if (a == "--sample")   sample = (uint32_t)std::stoul(nxt());
        else if (a == "--seed")     seed = (uint32_t)std::stoul(nxt());
        else if (a == "--rows")     cli_rows = (uint32_t)std::stoul(nxt());
        else if (a == "--cols")     cli_cols = (uint32_t)std::stoul(nxt());
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (corpus_path.empty() || cb_path.empty()) {
        std::fprintf(stderr, "usage: metrics --corpus c.sbcsr --codebook m --format wts|somw "
                             "[--sample N --seed S --rows R --cols C]\n");
        return 2;
    }

    Corpus c = load_sbcsr(corpus_path);
    uint32_t K = 0, V = 0, rows = 0, cols = 0;
    std::vector<float> W = load_codebook(cb_path, fmt, K, V, rows, cols, cli_rows, cli_cols);
    if (V != c.n_features)
        std::fprintf(stderr, "WARNING: codebook features %u != corpus features %u\n", V, c.n_features);

    // Per-neuron L2 norm ||w_k|| (feature-major access is contiguous over k for fixed v).
    std::vector<double> norm2(K, 0.0);
    for (uint32_t v = 0; v < V; ++v) {
        const float* wv = &W[(size_t)v * K];
        for (uint32_t k = 0; k < K; ++k) norm2[k] += (double)wv[k] * wv[k];
    }
    std::vector<float> norm(K);
    for (uint32_t k = 0; k < K; ++k) norm[k] = (float)std::sqrt(norm2[k]);

    // Choose the evaluation sample set (all, or a seeded random subset for speed).
    std::vector<uint32_t> idx;
    if (sample == 0 || sample >= c.n_samples) {
        idx.resize(c.n_samples);
        for (uint32_t i = 0; i < c.n_samples; ++i) idx[i] = i;
    } else {
        idx.resize(c.n_samples);
        for (uint32_t i = 0; i < c.n_samples; ++i) idx[i] = i;
        std::mt19937 rng(seed);
        for (uint32_t i = 0; i < sample; ++i) {
            std::uniform_int_distribution<uint32_t> d(i, c.n_samples - 1);
            std::swap(idx[i], idx[d(rng)]);
        }
        idx.resize(sample);
    }

    double qe_sum = 0.0;
    long te_bad = 0, counted = 0;
    std::vector<char> hit(K, 0);              // neuron used as BMU1 by >=1 eval sample
    #pragma omp parallel
    {
        std::vector<float> acc(K);
        #pragma omp for reduction(+:qe_sum,te_bad,counted) schedule(dynamic, 256)
        for (long s = 0; s < (long)idx.size(); ++s) {
            uint32_t i = idx[s];
            uint32_t a = c.row_ptr[i], b = c.row_ptr[i + 1];
            uint32_t nnz = b - a;
            if (nnz == 0) continue;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (uint32_t p = a; p < b; ++p) {           // accumulate dot(sample, w_k) for all k
                uint32_t fv = c.col_idx[p];
                if (fv >= V) continue;                    // feature beyond codebook (index mismatch)
                const float* wv = &W[(size_t)fv * K];
                for (uint32_t k = 0; k < K; ++k) acc[k] += wv[k];
            }
            float snorm = std::sqrt((float)nnz);
            // Best and second-best cosine similarity = acc[k] / (snorm * norm[k]).
            int b1 = -1, b2 = -1; float s1 = -1e30f, s2 = -1e30f;
            for (uint32_t k = 0; k < K; ++k) {
                if (norm[k] <= 0.0f) continue;
                float sim = acc[k] / (snorm * norm[k]);
                if (sim > s1) { s2 = s1; b2 = b1; s1 = sim; b1 = (int)k; }
                else if (sim > s2) { s2 = sim; b2 = (int)k; }
            }
            if (b1 < 0) continue;
            qe_sum += 1.0 - (double)s1;                  // cosine distance to BMU1
            counted += 1;
            hit[b1] = 1;                                 // benign race: all writes store 1
            if (b2 >= 0) {
                int r1 = b1 / (int)cols, c1 = b1 % (int)cols;
                int r2 = b2 / (int)cols, c2 = b2 % (int)cols;
                int cheb = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
                if (cheb > 1) te_bad += 1;
            }
        }
    }
    double qe = counted ? qe_sum / counted : 0.0;
    double te = counted ? (double)te_bad / counted : 0.0;
    long used = 0;
    for (uint32_t k = 0; k < K; ++k) used += hit[k];
    double dead = K ? (double)(K - used) / K : 0.0;
    std::printf("{\"n_eval\": %ld, \"neurons\": %u, \"features\": %u, \"bmu1_used\": %ld, "
                "\"qe_cosine\": %.6f, \"topographic_error\": %.6f, \"dead_fraction\": %.6f}\n",
                counted, K, V, used, qe, te, dead);
    return 0;
}
