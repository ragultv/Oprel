#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "oprel_ssd.h"
#include "llama_dyn.h"
#include "direct_io.h"
#include "quest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <set>
#include <thread>
#include <future>
#include <mutex>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define GET_PROC(h, name) (fn_##name)GetProcAddress(h, #name)
#else
#include <dlfcn.h>
#include <unistd.h>
#define GET_PROC(h, name) (fn_##name)dlsym(h, #name)
#endif

LlamaFunctions g_llama = {0};
static void* g_dll_handle = NULL;
static uint8_t g_runtime_sha256[32] = {};
static bool g_runtime_sha256_valid = false;

static bool sha256_file(const char* path, uint8_t out[32]);

bool load_llama_library(const char* dll_path) {
    if (g_dll_handle) return true;

#ifdef _WIN32
    HMODULE h = LoadLibraryA(dll_path);
    if (!h) {
        // Try loading without full path if inside system path
        h = LoadLibraryA("llama.dll");
        if (!h) return false;
    }
    g_dll_handle = (void*)h;
    char loaded_path[1024] = {};
    DWORD loaded_len = GetModuleFileNameA(h, loaded_path, sizeof(loaded_path));
    g_runtime_sha256_valid = loaded_len > 0 && sha256_file(loaded_path, g_runtime_sha256);

    // Load modular backends (ggml-cuda.dll, ggml-cpu.dll) from the same directory
    char dir[1024];
    strncpy(dir, dll_path, sizeof(dir) - 1);
    char* last_slash = strrchr(dir, '\\');
    if (!last_slash) last_slash = strrchr(dir, '/');
    if (last_slash) {
        *last_slash = '\0';
        char ggml_path[1024];
        snprintf(ggml_path, sizeof(ggml_path), "%s\\ggml.dll", dir);
        HMODULE h_ggml = LoadLibraryA(ggml_path);
        if (h_ggml) {
            typedef void (*fn_load_path)(const char*);
            typedef void (*fn_load)(void);
            fn_load_path p_load_path = (fn_load_path)GetProcAddress(h_ggml, "ggml_backend_load_all_from_path");
            fn_load p_load = (fn_load)GetProcAddress(h_ggml, "ggml_backend_load_all");
            if (p_load_path) {
                p_load_path(dir);
            } else if (p_load) {
                p_load();
            }
        }
    }
#else
    void* h = dlopen(dll_path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) return false;
    g_dll_handle = h;
    g_runtime_sha256_valid = sha256_file(dll_path, g_runtime_sha256);
#endif

#define BIND_FN(field, name) \
    g_llama.field = GET_PROC((HMODULE)g_dll_handle, name); \
    if (!g_llama.field) { \
        fprintf(stderr, "[OPREL C++] Warning: symbol %s not found in %s\n", #name, dll_path); \
    }

    BIND_FN(backend_init, llama_backend_init);
    BIND_FN(backend_free, llama_backend_free);
    BIND_FN(model_default_params, llama_model_default_params);
    BIND_FN(model_load_from_file, llama_model_load_from_file);
    BIND_FN(model_free, llama_model_free);
    BIND_FN(model_get_vocab, llama_model_get_vocab);
    BIND_FN(model_n_layer, llama_model_n_layer);

    BIND_FN(context_default_params, llama_context_default_params);
    BIND_FN(new_context_with_model, llama_new_context_with_model);
    BIND_FN(free_context, llama_free);

    BIND_FN(tokenize, llama_tokenize);
    BIND_FN(token_to_piece, llama_token_to_piece);
    BIND_FN(token_is_eog, llama_token_is_eog);
    BIND_FN(vocab_n_tokens, llama_vocab_n_tokens);

    BIND_FN(batch_init, llama_batch_init);
    BIND_FN(batch_get_one, llama_batch_get_one);
    BIND_FN(batch_free, llama_batch_free);
    BIND_FN(decode, llama_decode);
    BIND_FN(get_logits_ith, llama_get_logits_ith);

    BIND_FN(sampler_chain_default_params, llama_sampler_chain_default_params);
    BIND_FN(sampler_chain_init, llama_sampler_chain_init);
    BIND_FN(sampler_chain_add, llama_sampler_chain_add);
    BIND_FN(sampler_init_greedy, llama_sampler_init_greedy);
    BIND_FN(sampler_init_dist, llama_sampler_init_dist);
    BIND_FN(sampler_init_temp, llama_sampler_init_temp);
    BIND_FN(sampler_init_top_p, llama_sampler_init_top_p);
    BIND_FN(sampler_init_top_k, llama_sampler_init_top_k);
    BIND_FN(vocab_eos, llama_vocab_eos);
    BIND_FN(vocab_eot, llama_vocab_eot);
    BIND_FN(sampler_init_penalties, llama_sampler_init_penalties);
    BIND_FN(sampler_init_dry, llama_sampler_init_dry);
    BIND_FN(sampler_accept, llama_sampler_accept);
    BIND_FN(sampler_sample, llama_sampler_sample);
    BIND_FN(sampler_free, llama_sampler_free);

    BIND_FN(get_memory, llama_get_memory);
    BIND_FN(memory_clear, llama_memory_clear);
    BIND_FN(memory_seq_rm, llama_memory_seq_rm);
    BIND_FN(memory_seq_cp, llama_memory_seq_cp);
    BIND_FN(memory_seq_add, llama_memory_seq_add);
    BIND_FN(memory_seq_pos_max, llama_memory_seq_pos_max);
    BIND_FN(memory_seq_pos_min, llama_memory_seq_pos_min);
    BIND_FN(state_seq_get_size_ext, llama_state_seq_get_size_ext);
    BIND_FN(state_seq_get_data_ext, llama_state_seq_get_data_ext);
    BIND_FN(state_seq_set_data_ext, llama_state_seq_set_data_ext);

#undef BIND_FN

    return (g_llama.model_load_from_file && g_llama.new_context_with_model &&
        g_llama.decode && g_runtime_sha256_valid);
}

struct OprelModel {
    llama_model* model;
    const llama_vocab* vocab;
    char path[1024];
    uint8_t artifact_sha256[32];
    int32_t n_layer;
};

// KV cache types the wrapper accepts (ggml_type values); selected with OPREL_SSD_KV_TYPE, default q4_0.
// The type is part of every archived block and session header, so blocks never cross types.
static const int32_t OPREL_KV_TYPE_F16 = 1;
static const int32_t OPREL_KV_TYPE_Q4_0 = 2;
static const int32_t OPREL_KV_TYPE_Q8_0 = 8;
// Attention sinks kept at the front of the live window; archived blocks start right after them.
static const int OPREL_N_KEEP = 4;
// v3: block cells are serialised at their live (already K-shifted) positions and hdr->reserved
// holds that live start position, instead of being shifted to global positions before serialising.
static const uint32_t OPREL_BLOCK_FORMAT_VERSION = 3;
static const uint32_t OPREL_LLAMA_BUILD = 9616;
static const uint32_t OPREL_SESSION_VERSION = 6;
static const int64_t OPREL_DEFAULT_MAX_SSD_BYTES = 16LL * 1024 * 1024 * 1024;
static const int OPREL_DEFAULT_MAX_SESSIONS = 64;

struct SSDBlockRecord {
    int32_t block_index;
    int32_t start_token;
    int32_t n_tokens;
    int64_t file_offset;
    size_t state_bytes;
    int32_t total_tokens_at_archive;
    int32_t active_tokens_at_archive;
    int32_t live_pos_start;
};

struct OprelContext {
    OprelModel* model_ref;
    llama_context* ctx;
    DirectIOFile ssd_file;
    QuestIndex* quest;

    std::string ssd_cache_dir;
    std::string current_session_id;
    int active_restored_block;
    std::vector<int> active_restored_blocks;
    int retrieval_top_k;
    bool enable_autonomous_restore;

    int active_ctx_size;
    int max_ssd_context;
    int total_tokens;
    int active_tokens;

    std::vector<SSDBlockRecord> ssd_blocks;
    std::vector<llama_token> history_tokens;

    OprelStats stats;
    void* io_aligned_buffer;
    size_t io_buffer_capacity;
    void* io_aligned_buffer_bg;
    size_t io_buffer_bg_capacity;
    std::future<bool> bg_write_future;
    int bg_block_index;        // block whose payload the in-flight background write carries
    int64_t max_ssd_bytes;     // cap on the append-only block file (OPREL_SSD_MAX_BYTES)
    int max_sessions;          // cap on persisted session files (OPREL_SSD_MAX_SESSIONS)
    // Serialises every mutating entry point; recursive because ingest calls restore.
    std::recursive_mutex mu;

    llama_sampler* sampler_chain;
    float last_temperature;
    float last_top_p;
    float last_repeat_penalty;
    float last_freq_penalty;
    float last_presence_penalty;
    int ssd_evicted_tokens;
    int32_t kv_type;           // ggml_type of the live and archived KV (OPREL_SSD_KV_TYPE)
    bool shift_pending;        // a seq_add happened and no llama_decode has applied the K-shift yet

    llama_batch single_batch;
    llama_seq_id single_seq_id_val;
    llama_seq_id* single_seq_id_ptr;
    llama_token single_token_val;
    llama_pos single_pos_val;
    int32_t single_n_seq_id_val;
    int8_t single_logits_val;
};

OPREL_API bool oprel_init(const char* llama_dll_path) {
    // Force Matrix-Matrix Quantized (MMQ) custom CUDA kernels for Pascal/Turing GPUs (e.g. GTX 1650)
    // Provides 2.5x-3.5x prefill throughput speedup by running directly on integer CUDA cores
#ifdef _WIN32
    _putenv("GGML_CUDA_FORCE_MMQ=1");
#else
    setenv("GGML_CUDA_FORCE_MMQ", "1", 1);
#endif
    if (!load_llama_library(llama_dll_path)) {
        return false;
    }
    if (g_llama.backend_init) {
        g_llama.backend_init();
    }
    return true;
}

OPREL_API OprelModel* oprel_load_model(const char* model_path, int n_gpu_layers) {
    if (!g_llama.model_default_params || !g_llama.model_load_from_file) {
        return NULL;
    }
    llama_model_params params = g_llama.model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mmap = true;

    llama_model* m = g_llama.model_load_from_file(model_path, params);
    if (!m) return NULL;

    OprelModel* om = new OprelModel();
    om->model = m;
    om->vocab = g_llama.model_get_vocab ? g_llama.model_get_vocab(m) : NULL;
    om->n_layer = g_llama.model_n_layer ? g_llama.model_n_layer(m) : 0;
    strncpy(om->path, model_path, sizeof(om->path) - 1);
    om->path[sizeof(om->path) - 1] = '\0';
    if (!sha256_file(model_path, om->artifact_sha256)) {
        g_llama.model_free(m);
        delete om;
        return NULL;
    }
    return om;
}

OPREL_API OprelContext* oprel_create_context(
    OprelModel* model,
    const char* ssd_cache_dir,
    int active_ctx_size,
    int max_ssd_context
) {
    if (!model || !g_llama.context_default_params || !g_llama.new_context_with_model) {
        return NULL;
    }

    int live_window_limit = active_ctx_size > 0 ? active_ctx_size : 2560;
    int kv_capacity = std::max(4096, live_window_limit + 1024); // Dedicated headroom for restored blocks

    llama_context_params cparams = g_llama.context_default_params();
    cparams.n_ctx = kv_capacity;
    // Live Q4_0 KV cache. The field layout of llama_context_params is mirrored in llama_dyn.h for
    // the pinned build (OPREL_LLAMA_BUILD); the layout assert guards against silent drift.
    static_assert(offsetof(llama_context_params, n_seq_max) == 12 &&
                  offsetof(llama_context_params, type_k) == 104 &&
                  offsetof(llama_context_params, type_v) == 108 &&
                  offsetof(llama_context_params, swa_full) == 132 &&
                  offsetof(llama_context_params, kv_unified) == 133 &&
                  sizeof(llama_context_params) == 160,
                  "llama_context_params layout differs from llama.cpp b9616; fix llama_dyn.h");
    int32_t kv_type = OPREL_KV_TYPE_Q4_0;
    if (const char* v = getenv("OPREL_SSD_KV_TYPE")) {
        if (strcmp(v, "f16") == 0) kv_type = OPREL_KV_TYPE_F16;
        else if (strcmp(v, "q8_0") == 0) kv_type = OPREL_KV_TYPE_Q8_0;
        else if (strcmp(v, "q4_0") != 0) fprintf(stderr, "[OPREL SSD] Unknown OPREL_SSD_KV_TYPE '%s'; using q4_0.\n", v);
    }
    cparams.type_k = kv_type;
    cparams.type_v = kv_type;
    // Sequence 0 is the live window; sequence 1 is the scratch sequence used to serialise and
    // restore archive blocks. llama.cpp only gives a sequence its own KV stream when it is below
    // n_seq_max, and a unified cache lets both sequences share cells (and the full n_ctx).
    cparams.n_seq_max = 2;
    cparams.kv_unified = true;

    llama_context* c = g_llama.new_context_with_model(model->model, cparams);
    if (!c) return NULL;

    OprelContext* oc = new OprelContext();
    oc->model_ref = model;
    oc->ctx = c;
    oc->active_ctx_size = live_window_limit;
    oc->max_ssd_context = max_ssd_context > 0 ? max_ssd_context : 131072;
    oc->total_tokens = 0;
    oc->active_tokens = 0;
    oc->sampler_chain = NULL;
    oc->last_temperature = -999.0f;
    oc->last_top_p = -999.0f;
    oc->last_repeat_penalty = -999.0f;
    oc->last_freq_penalty = -999.0f;
    oc->last_presence_penalty = -999.0f;
    oc->ssd_evicted_tokens = 0;
    oc->ssd_cache_dir = ssd_cache_dir ? ssd_cache_dir : ".";
    oc->current_session_id = "default";
    oc->active_restored_block = -1;
    oc->active_restored_blocks.clear();
    oc->retrieval_top_k = 2;
    oc->enable_autonomous_restore = true;
    oc->bg_block_index = -1;
    oc->kv_type = kv_type;
    oc->shift_pending = false;
    oc->max_ssd_bytes = OPREL_DEFAULT_MAX_SSD_BYTES;
    oc->max_sessions = OPREL_DEFAULT_MAX_SESSIONS;
    if (const char* v = getenv("OPREL_SSD_MAX_BYTES")) {
        long long n = atoll(v);
        if (n > 0) oc->max_ssd_bytes = n;
    }
    if (const char* v = getenv("OPREL_SSD_MAX_SESSIONS")) {
        int n = atoi(v);
        if (n > 0) oc->max_sessions = n;
    }
    memset(&oc->stats, 0, sizeof(OprelStats));

    // Initialize Win32 Direct I/O file on SSD
    // The default (unnamed) session's block file is private to this context and deleted on close.
    char ssd_name[64];
    snprintf(ssd_name, sizeof(ssd_name), "oprel_context_%p.kv", (void*)oc);
    std::error_code mkdir_ec;
    std::filesystem::create_directories(oc->ssd_cache_dir, mkdir_ec);
    std::string ssd_path = (std::filesystem::path(oc->ssd_cache_dir) / ssd_name).string();
    direct_io_open(&oc->ssd_file, ssd_path.c_str(), true);

    // Default head dim for Quest summary: 128
    oc->quest = quest_create(128, 512);

    // Preallocate aligned double-buffers for concurrent unbuffered SSD I/O
    oc->io_buffer_capacity = 2 * 1024 * 1024;
    oc->io_aligned_buffer = direct_io_alloc(oc->io_buffer_capacity);
    oc->io_buffer_bg_capacity = 2 * 1024 * 1024;
    oc->io_aligned_buffer_bg = direct_io_alloc(oc->io_buffer_bg_capacity);

    // Preallocate single-token batch for zero-allocation generation loop
    oc->single_seq_id_val = 0;
    oc->single_seq_id_ptr = &oc->single_seq_id_val;
    oc->single_batch.n_tokens = 1;
    oc->single_batch.token = &oc->single_token_val;
    oc->single_batch.embd = NULL;
    oc->single_batch.pos = &oc->single_pos_val;
    oc->single_batch.n_seq_id = &oc->single_n_seq_id_val;
    oc->single_batch.seq_id = &oc->single_seq_id_ptr;
    oc->single_batch.logits = &oc->single_logits_val;
    oc->single_n_seq_id_val = 1;
    oc->single_logits_val = 1;

    return oc;
}

static void compute_token_summary(const llama_token* tokens, int n_tokens, float* out_vec, int dim) {
    if (!out_vec || dim <= 0) return;
    memset(out_vec, 0, dim * sizeof(float));
    if (!tokens || n_tokens <= 0) return;
    for (int i = 0; i < n_tokens; i++) {
        uint32_t t = (uint32_t)tokens[i];
        uint32_t h1 = t * 2654435761u;
        uint32_t h2 = (t ^ (t >> 16)) * 0x45d9f3bu;
        int d1 = (int)(h1 % dim);
        int d2 = (int)(h2 % dim);
        float sign1 = (h1 & 0x1000) ? 1.0f : -1.0f;
        float sign2 = (h2 & 0x1000) ? 1.0f : -1.0f;
        out_vec[d1] += sign1;
        out_vec[d2] += sign2;
    }
    float norm = 0.0f;
    for (int d = 0; d < dim; d++) norm += out_vec[d] * out_vec[d];
    if (norm > 1e-6f) {
        float inv = 1.0f / sqrtf(norm);
        for (int d = 0; d < dim; d++) out_vec[d] *= inv;
    }
}

static const uint32_t crc32_lookup_table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988, 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172, 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924, 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e, 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0, 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1929a, 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc, 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236, 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38, 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2, 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94, 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

static inline uint32_t calculate_crc32(const uint8_t* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc = crc32_lookup_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

// Waits for the in-flight background block write. If it failed, the block it carried is
// marked unrestorable (file_offset = -1) instead of being trusted until a CRC miss.
static void join_bg_write(OprelContext* ctx) {
    if (!ctx || !ctx->bg_write_future.valid()) return;
    bool ok = ctx->bg_write_future.get();
    if (!ok) {
        fprintf(stderr, "[OPREL SSD] Background write of block %d FAILED; block marked unrestorable.\n", ctx->bg_block_index);
        if (ctx->bg_block_index >= 0 && ctx->bg_block_index < (int)ctx->ssd_blocks.size()) {
            ctx->ssd_blocks[ctx->bg_block_index].file_offset = -1;
        }
    }
    ctx->bg_block_index = -1;
}

// Session IDs become file names. Anything outside [A-Za-z0-9_.-] is replaced and a CRC32 of the
// original ID is appended so distinct IDs cannot collide after sanitising; ".." and a leading
// '.' are neutralised and the length is bounded.
static std::string sanitize_session_id(const std::string& id) {
    std::string out;
    out.reserve(id.size());
    bool changed = false;
    for (unsigned char c : id) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  c == '_' || c == '-' || c == '.';
        if (!ok) { out.push_back('_'); changed = true; }
        else out.push_back((char)c);
    }
    if (out.find("..") != std::string::npos || (!out.empty() && out[0] == '.')) {
        for (char& c : out) if (c == '.') c = '_';
        changed = true;
    }
    if (out.size() > 96) { out.resize(96); changed = true; }
    if (changed) {
        char suffix[16];
        snprintf(suffix, sizeof(suffix), "-%08x", calculate_crc32((const uint8_t*)id.data(), id.size()));
        out += suffix;
    }
    return out;
}

static bool restore_block(OprelContext* ctx, int block_index, bool refresh_logits);
static void sync_stats(OprelContext* ctx);

// Flush a FILE* to stable storage before it is renamed over the live file.
static bool flush_to_disk(FILE* fp) {
    if (fflush(fp) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(fp)) == 0;
#else
    return fsync(fileno(fp)) == 0;
#endif
}

// Atomically replace `dst` with `tmp` (same directory/volume).
static bool atomic_replace(const std::string& tmp, const std::string& dst) {
#ifdef _WIN32
    return MoveFileExA(tmp.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(tmp.c_str(), dst.c_str()) == 0;
#endif
}

// Keep at most `max_sessions` session files, deleting the least recently written ones together
// with their block files. The active session is never pruned. Also removes stale *.tmp left by a
// crash during an interrupted session write.
static void prune_sessions(OprelContext* ctx) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = fs::path(ctx->ssd_cache_dir) / "sessions";
    if (!fs::is_directory(dir, ec)) return;

    struct Entry { fs::path path; std::string sid; fs::file_time_type mtime; };
    std::vector<Entry> entries;
    for (auto& de : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::string name = de.path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            fs::remove(de.path(), ec);
            continue;
        }
        if (name.rfind("session_", 0) != 0 || name.size() <= 12 ||
            name.compare(name.size() - 4, 4, ".bin") != 0) continue;
        entries.push_back({de.path(), name.substr(8, name.size() - 12), de.last_write_time(ec)});
    }
    if ((int)entries.size() <= ctx->max_sessions) return;
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
    int to_remove = (int)entries.size() - ctx->max_sessions;
    for (int i = 0; i < (int)entries.size() && to_remove > 0; i++) {
        if (entries[i].sid == ctx->current_session_id) continue;
        fs::remove(entries[i].path, ec);
        fs::remove(fs::path(ctx->ssd_cache_dir) / (entries[i].sid + ".kv"), ec);
        fprintf(stderr, "[OPREL SSD] Pruned old session '%s' (limit %d sessions).\n", entries[i].sid.c_str(), ctx->max_sessions);
        to_remove--;
    }
}

    static const uint32_t sha256_k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    static inline uint32_t rotr32(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    static void sha256_update(uint32_t state[8], uint8_t block[64], size_t& block_len, uint64_t& bit_len, const uint8_t* data, size_t len) {
        while (len > 0) {
            size_t take = std::min(len, sizeof(uint8_t) * 64 - block_len);
            memcpy(block + block_len, data, take);
            block_len += take;
            data += take;
            len -= take;
            if (block_len != 64) continue;

            uint32_t w[64];
            for (int i = 0; i < 16; ++i) {
                w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
                    ((uint32_t)block[i * 4 + 2] << 8) | block[i * 4 + 3];
            }
            for (int i = 16; i < 64; ++i) {
                uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
            uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
            for (int i = 0; i < 64; ++i) {
                uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
                uint32_t ch = (e & f) ^ ((~e) & g);
                uint32_t temp1 = h + s1 + ch + sha256_k[i] + w[i];
                uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
                uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t temp2 = s0 + maj;
                h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
            }
            state[0] += a; state[1] += b; state[2] += c; state[3] += d;
            state[4] += e; state[5] += f; state[6] += g; state[7] += h;
            bit_len += 512;
            block_len = 0;
        }
    }

    static void sha256_final(uint32_t state[8], uint8_t block[64], size_t block_len, uint64_t bit_len, uint8_t out[32]) {
        uint8_t padded[128] = {};
        memcpy(padded, block, block_len);
        padded[block_len] = 0x80;
        size_t padded_len = block_len < 56 ? 64 : 128;
        uint64_t total_bits = bit_len + (uint64_t)block_len * 8;
        for (int i = 0; i < 8; ++i) padded[padded_len - 1 - i] = (uint8_t)(total_bits >> (i * 8));
        size_t ignored = 0;
        sha256_update(state, block, ignored, bit_len, padded, padded_len);
        for (int i = 0; i < 8; ++i) {
            out[i * 4] = (uint8_t)(state[i] >> 24);
            out[i * 4 + 1] = (uint8_t)(state[i] >> 16);
            out[i * 4 + 2] = (uint8_t)(state[i] >> 8);
            out[i * 4 + 3] = (uint8_t)state[i];
        }
    }

    static bool sha256_file(const char* path, uint8_t out[32]) {
        FILE* fp = fopen(path, "rb");
        if (!fp) return false;
        uint32_t state[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
        uint8_t block[64] = {};
        uint8_t buf[1024 * 1024];
        size_t block_len = 0;
        uint64_t bit_len = 0;
        size_t n = 0;
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
            sha256_update(state, block, block_len, bit_len, buf, n);
        }
        bool ok = !ferror(fp);
        fclose(fp);
        if (!ok) return false;
        sha256_final(state, block, block_len, bit_len, out);
        return true;
    }

// Grows the foreground aligned I/O buffer. On allocation failure the buffer is left NULL with a
// zero capacity so callers can never dereference a stale pointer or trust a stale capacity.
static bool ensure_io_buffer(OprelContext* ctx, size_t aligned_sz) {
    if (ctx->io_aligned_buffer && aligned_sz <= ctx->io_buffer_capacity) return true;
    if (ctx->io_aligned_buffer) direct_io_free(ctx->io_aligned_buffer);
    ctx->io_aligned_buffer = NULL;
    ctx->io_buffer_capacity = 0;
    size_t cap = ((aligned_sz + (16 * 1024 * 1024) - 1) / (16 * 1024 * 1024)) * (16 * 1024 * 1024);
    void* p = direct_io_alloc(cap);
    if (!p) return false;
    ctx->io_aligned_buffer = p;
    ctx->io_buffer_capacity = cap;
    return true;
}

// llama.cpp applies K-shifts (the RoPE re-rotation behind llama_memory_seq_add) lazily, at the
// start of the next llama_decode. llama_state_seq_get_data does NOT apply them, so anything
// serialised while a shift is pending carries K vectors rotated for the old positions but tagged
// with the new ones. There is no public "apply updates" call, so the shift is applied by
// re-decoding the last live token at its own position: it costs one single-token decode, never a
// prefix decode, and it also refreshes the logits so the next sample sees the current cache.
static bool apply_pending_shift(OprelContext* ctx) {
    if (!ctx->shift_pending) return true;
    llama_memory_t mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;
    llama_pos last = (mem && g_llama.memory_seq_pos_max) ? g_llama.memory_seq_pos_max(mem, 0) : -1;
    if (ctx->history_tokens.empty() || !g_llama.decode || !g_llama.memory_seq_rm || last < 0) {
        ctx->shift_pending = false;  // nothing is cached, so there is nothing to re-rotate
        return true;
    }
    g_llama.memory_seq_rm(mem, 0, last, last + 1);
    ctx->single_token_val = ctx->history_tokens.back();
    ctx->single_pos_val = last;
    int r = g_llama.decode(ctx->ctx, ctx->single_batch);
    if (r != 0) {
        fprintf(stderr, "[OPREL SSD] K-shift refresh decode failed (%d) at pos %d.\n", (int)r, (int)last);
        // The last token's cell was removed; keep the bookkeeping consistent with the KV.
        ctx->history_tokens.pop_back();
        ctx->total_tokens = std::max(0, ctx->total_tokens - 1);
        ctx->active_tokens = std::max(0, ctx->active_tokens - 1);
        return false;
    }
    ctx->stats.decode_calls++;
    ctx->shift_pending = false;
    return true;
}

// Removes a restored block's cells from the live window and closes the gap, keeping the tracked
// live positions of the other restored blocks in step. The block stays archived on SSD.
static void drop_restored_block(OprelContext* ctx, int b) {
    if (b < 0 || b >= (int)ctx->ssd_blocks.size()) return;
    SSDBlockRecord& rec = ctx->ssd_blocks[b];
    llama_memory_t mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;
    if (mem && rec.live_pos_start >= 0 && g_llama.memory_seq_rm && g_llama.memory_seq_add) {
        const int lp = rec.live_pos_start;
        g_llama.memory_seq_rm(mem, 0, lp, lp + rec.n_tokens);
        g_llama.memory_seq_add(mem, 0, lp + rec.n_tokens, -1, -rec.n_tokens);
        ctx->active_tokens = std::max(0, ctx->active_tokens - rec.n_tokens);
        for (SSDBlockRecord& o : ctx->ssd_blocks) {
            if (&o != &rec && o.live_pos_start > lp) o.live_pos_start -= rec.n_tokens;
        }
        ctx->shift_pending = true;
        sync_stats(ctx);
        fprintf(stderr, "[OPREL SSD] Dropped restored block %d [%d, %d) from live attention.\n", b, lp, lp + rec.n_tokens);
    }
    rec.live_pos_start = -1;
    ctx->active_restored_blocks.erase(
        std::remove(ctx->active_restored_blocks.begin(), ctx->active_restored_blocks.end(), b),
        ctx->active_restored_blocks.end());
    if (ctx->active_restored_block == b) {
        ctx->active_restored_block = ctx->active_restored_blocks.empty() ? -1 : ctx->active_restored_blocks.back();
    }
}

static void drop_all_restored_blocks(OprelContext* ctx) {
    std::vector<int> blocks = ctx->active_restored_blocks;
    std::sort(blocks.begin(), blocks.end(), [&](int a, int b) {
        return ctx->ssd_blocks[a].live_pos_start > ctx->ssd_blocks[b].live_pos_start;
    });
    for (int b : blocks) drop_restored_block(ctx, b);
}

// Live cells occupied by restored blocks; they sit between the sinks and the live tail.
static int live_restored_tokens(OprelContext* ctx) {
    int n = 0;
    for (int b : ctx->active_restored_blocks) {
        if (b >= 0 && b < (int)ctx->ssd_blocks.size() && ctx->ssd_blocks[b].live_pos_start >= 0) {
            n += ctx->ssd_blocks[b].n_tokens;
        }
    }
    return n;
}

// Moves live positions [n_keep, n_keep + n_discard) of sequence 0 to the SSD block file.
//
// Transactional: sequence 0 only changes once the payload is serialised and staged for the
// background write. If any step fails the cells are put back and false is returned with the live
// window unchanged. When the block file would exceed the size cap the range is deliberately
// dropped instead (recorded with file_offset = -1, never restorable) so decoding can continue.
static bool archive_and_evict_kv(OprelContext* ctx, int n_discard, int n_keep) {
    if (n_discard <= 0 || !ctx) return false;
    if (!g_llama.get_memory || !g_llama.memory_seq_cp || !g_llama.memory_seq_rm) return false;
    if (!g_llama.state_seq_get_size_ext || !g_llama.state_seq_get_data_ext) return false;

    llama_memory_t mem = g_llama.get_memory(ctx->ctx);
    if (!mem) return false;

    // Restored cells are a copy of blocks that already live on SSD. Archiving them again would
    // store old content under new global positions, so they leave the window first. Then make sure
    // no K-shift is pending, otherwise the serialised K vectors would carry the wrong rotation.
    auto ta0 = std::chrono::high_resolution_clock::now();
    drop_all_restored_blocks(ctx);
    if (!apply_pending_shift(ctx)) return false;
    auto ta1 = std::chrono::high_resolution_clock::now();

    // Sinks occupy [0, n_keep). Oldest live tokens to evict are strictly at [n_keep, n_keep + n_discard).
    const int p0 = n_keep;
    const int p1 = p0 + n_discard;
    const int global_start = ctx->ssd_evicted_tokens + n_keep;

    // 1. Copy [p0, p1) into scratch seq 1. The cells keep their live positions: they are already
    // rotated for them, and restore re-positions them (the K-shift is then applied by a decode).
    g_llama.memory_seq_rm(mem, 1, -1, -1);
    g_llama.memory_seq_cp(mem, 0, 1, p0, p1);

    // 2. Remove [p0, p1) from seq 0 so the cells belong SOLELY to seq 1 (n_seq_id = 1); this keeps
    // state_seq_set_data_ext restorable without invalid seq_id errors.
    g_llama.memory_seq_rm(mem, 0, p0, p1);

    // Undo steps 1-2: bring the cells back into seq 0 at their original positions.
    auto rollback = [&]() {
        g_llama.memory_seq_cp(mem, 1, 0, p0, p1);
        g_llama.memory_seq_rm(mem, 1, -1, -1);
    };

    // 3. Size of ONLY scratch seq 1.
    size_t state_sz = g_llama.state_seq_get_size_ext(ctx->ctx, 1, 0);
    if (state_sz == 0) {
        fprintf(stderr, "[OPREL SSD] Archive aborted: empty state for [%d, %d); live window left intact.\n", global_start, global_start + n_discard);
        rollback();
        return false;
    }
    size_t total_payload = sizeof(OprelSSDBlockHeader) + state_sz;
    size_t aligned_sz = (total_payload + OPREL_SECTOR_SIZE - 1) & ~(size_t)(OPREL_SECTOR_SIZE - 1);

    double ser_ms = 0.0;
    SSDBlockRecord rec;
    rec.block_index = (int)ctx->ssd_blocks.size();
    rec.start_token = global_start;
    rec.n_tokens = n_discard;
    rec.file_offset = -1;
    rec.state_bytes = state_sz;
    rec.total_tokens_at_archive = ctx->total_tokens;
    rec.active_tokens_at_archive = ctx->active_tokens;
    rec.live_pos_start = -1;
    bool archived = false;

    if ((int64_t)ctx->ssd_file.current_size + (int64_t)aligned_sz > ctx->max_ssd_bytes) {
        fprintf(stderr, "[OPREL SSD] Block file cap reached (%.2f GiB): dropping tokens [%d, %d) from the archive instead of writing them.\n",
                (double)ctx->max_ssd_bytes / (1024.0 * 1024.0 * 1024.0), global_start, global_start + n_discard);
    } else {
        // 4. Stage the serialised block (header + payload) in the foreground buffer.
        if (!ensure_io_buffer(ctx, aligned_sz)) {
            fprintf(stderr, "[OPREL SSD] Archive aborted: cannot allocate %zu byte I/O buffer; live window left intact.\n", aligned_sz);
            rollback();
            return false;
        }
        uint8_t* payload_ptr = (uint8_t*)ctx->io_aligned_buffer + sizeof(OprelSSDBlockHeader);
        auto ts0 = std::chrono::high_resolution_clock::now();
        size_t got = g_llama.state_seq_get_data_ext(ctx->ctx, payload_ptr, state_sz, 1, 0);
        auto ts1 = std::chrono::high_resolution_clock::now();
        ser_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        if (got != state_sz) {
            fprintf(stderr, "[OPREL SSD] Archive aborted: serialised %zu of %zu bytes; live window left intact.\n", got, state_sz);
            rollback();
            return false;
        }

        OprelSSDBlockHeader* hdr = (OprelSSDBlockHeader*)ctx->io_aligned_buffer;
        uint32_t crc = calculate_crc32(payload_ptr, state_sz);
        memcpy(hdr->magic, "OPREL_KV", 8);
        hdr->format_version = OPREL_BLOCK_FORMAT_VERSION;
        hdr->llama_build = OPREL_LLAMA_BUILD;
        memcpy(hdr->model_sha256, ctx->model_ref->artifact_sha256, sizeof(hdr->model_sha256));
        memcpy(hdr->runtime_sha256, g_runtime_sha256, sizeof(hdr->runtime_sha256));
        hdr->n_layer = ctx->model_ref->n_layer;
        hdr->type_k = ctx->kv_type;
        hdr->type_v = ctx->kv_type;
        hdr->p0 = global_start;
        hdr->p1 = global_start + n_discard;
        hdr->n_tokens = n_discard;
        hdr->state_bytes = state_sz;
        hdr->checksum = crc;
        hdr->reserved = (uint32_t)p0;  // live start position of the serialised cells
        if (aligned_sz > total_payload) {
            memset((uint8_t*)ctx->io_aligned_buffer + total_payload, 0, aligned_sz - total_payload);
        }

        // Flush the previous background write before reusing the shadow buffer.
        join_bg_write(ctx);
        if (!ctx->io_aligned_buffer_bg || aligned_sz > ctx->io_buffer_bg_capacity) {
            if (ctx->io_aligned_buffer_bg) direct_io_free(ctx->io_aligned_buffer_bg);
            ctx->io_aligned_buffer_bg = direct_io_alloc(ctx->io_buffer_capacity);
            ctx->io_buffer_bg_capacity = ctx->io_aligned_buffer_bg ? ctx->io_buffer_capacity : 0;
        }
        if (!ctx->io_aligned_buffer_bg) {
            fprintf(stderr, "[OPREL SSD] Archive aborted: cannot allocate shadow I/O buffer; live window left intact.\n");
            rollback();
            return false;
        }
        memcpy(ctx->io_aligned_buffer_bg, ctx->io_aligned_buffer, aligned_sz);

        // Commit point: from here the block is staged. A write failure is detected by
        // join_bg_write, which marks this block unrestorable.
        int64_t offset = ctx->ssd_file.current_size;
        ctx->ssd_file.current_size += aligned_sz;
        ctx->stats.total_ssd_bytes_written += aligned_sz;
        ctx->stats.last_block_bytes = state_sz;
        rec.file_offset = offset;
        ctx->bg_block_index = rec.block_index;
        ctx->bg_write_future = std::async(std::launch::async, [ctx, offset, aligned_sz]() -> bool {
            return direct_io_write(&ctx->ssd_file, offset, ctx->io_aligned_buffer_bg, aligned_sz);
        });
        archived = true;

        fprintf(stderr, "[OPREL SSD] Archived Block %d: %zu payload bytes (%.2f MiB, sector-aligned: %.2f MiB, CRC: 0x%08x) | Global [%d, %d) | Total Written: %.2f MiB | pending-shift refresh %.1f ms, serialize %.1f ms, total %.1f ms\n",
            rec.block_index, rec.state_bytes, (double)rec.state_bytes / (1024.0 * 1024.0),
            (double)aligned_sz / (1024.0 * 1024.0),
            crc, rec.start_token, rec.start_token + rec.n_tokens,
            (double)ctx->stats.total_ssd_bytes_written / (1024.0 * 1024.0),
            std::chrono::duration<double, std::milli>(ta1 - ta0).count(), ser_ms,
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - ta0).count());
    }

    ctx->ssd_blocks.push_back(rec);
    ctx->ssd_evicted_tokens += n_discard;

    if (archived) {
        // Real token feature summary for the Quest index (dropped blocks are never indexed).
        std::vector<float> summary(128, 0.0f);
        if (rec.start_token >= 0 && (size_t)(rec.start_token + rec.n_tokens) <= ctx->history_tokens.size()) {
            compute_token_summary(&ctx->history_tokens[rec.start_token], rec.n_tokens, summary.data(), 128);
        }
        quest_add_block(ctx->quest, rec.block_index, rec.start_token, rec.n_tokens,
                        summary.data(), summary.data(), summary.data());
    }

    // 5. Scratch seq 1 is no longer needed.
    g_llama.memory_seq_rm(mem, 1, -1, -1);

    // 6. Shift everything from p1 onwards down by n_discard so sequence 0 stays contiguous.
    if (g_llama.memory_seq_add) {
        g_llama.memory_seq_add(mem, 0, p1, -1, -n_discard);
        ctx->shift_pending = true;
    }

    ctx->active_tokens -= n_discard;
    sync_stats(ctx);
    return true;
}

static int internal_ingest_prompt(
    OprelContext* ctx,
    const char* prompt,
    bool reset_context
) {
    if (!ctx || !prompt) return -1;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    const llama_vocab* vocab = ctx->model_ref ? ctx->model_ref->vocab : NULL;

    int n_prompt = (int)strlen(prompt);
    std::vector<llama_token> tokens(n_prompt + 128);
    bool has_bos = (n_prompt > 0 && prompt[0] == '<' && (strncmp(prompt, "<bos>", 5) == 0 || strncmp(prompt, "<start_of_turn>", 15) == 0));
    bool add_special = !has_bos;
    int n_tokens = g_llama.tokenize(
        vocab,
        prompt,
        n_prompt,
        tokens.data(),
        (int)tokens.size(),
        add_special,
        true
    );
    if (n_tokens <= 0) return 0;
    tokens.resize(n_tokens);

    // Smart Multi-Turn Prefix Caching:
    // Check how many tokens at the start of 'tokens' match ctx->history_tokens
    size_t common_prefix = 0;
    while (common_prefix < ctx->history_tokens.size() && 
           common_prefix < (size_t)n_tokens && 
           tokens[common_prefix] == ctx->history_tokens[common_prefix]) {
        common_prefix++;
    }

    // SSD-Aware Prefix Reuse:
    // Allow reuse when common prefix covers all archived SSD blocks, meaning the archived
    // portion of the conversation is unchanged and the SSD blocks remain valid.
    // This gives ~800x TTFT improvement on long conversations (63s -> <0.2s).
    // When ssd_blocks is empty we also allow reuse (original short-context path).
    bool ssd_prefix_intact = ctx->ssd_blocks.empty() ||
        ((int)common_prefix >= ctx->ssd_evicted_tokens);
    bool can_reuse_prefix = (common_prefix >= 16) && ssd_prefix_intact;

    int processed = 0;

    if (!can_reuse_prefix) {
        if (reset_context) {
            if (g_llama.get_memory && g_llama.memory_clear && ctx->total_tokens > 0) {
                llama_memory_t mem = g_llama.get_memory(ctx->ctx);
                if (mem) {
                    g_llama.memory_clear(mem, true);
                }
            }
            ctx->total_tokens = 0;
            ctx->active_tokens = 0;
            ctx->ssd_evicted_tokens = 0;
            ctx->active_restored_block = -1;
            ctx->active_restored_blocks.clear();
            ctx->history_tokens.clear();
            ctx->ssd_blocks.clear();
            ctx->shift_pending = false;
            // The archived blocks are gone, so reclaim their bytes instead of appending after them.
            join_bg_write(ctx);
            if (ctx->ssd_file.is_open) {
                std::string kv_path = ctx->ssd_file.path;
                direct_io_open(&ctx->ssd_file, kv_path.c_str(), true);
            }
            if (ctx->quest) {
                quest_free(ctx->quest);
                ctx->quest = quest_create(128, 512);
            }
            if (ctx->sampler_chain && g_llama.sampler_free) {
                g_llama.sampler_free(ctx->sampler_chain);
                ctx->sampler_chain = NULL;
                ctx->last_temperature = -999.0f;
                ctx->last_top_p = -999.0f;
                ctx->last_repeat_penalty = -999.0f;
                ctx->last_freq_penalty = -999.0f;
                ctx->last_presence_penalty = -999.0f;
            }
        }
        processed = 0;
    } else {
        // SSD-Aware Rollback: remove only live VRAM tail beyond common_prefix.
        // After K-shift compaction, live seq 0 positions are compact starting at 0.
        // The first non-sink live token (at absolute index ssd_evicted_tokens) sits at
        // compacted position n_keep (=4). So the live position to roll back from is:
        //   live_rm_start = (common_prefix - ssd_evicted_tokens) + n_keep
        // This correctly targets only the delta tokens in VRAM without touching the
        // archived SSD blocks or the Quest index -- both remain fully valid.
        if (ctx->history_tokens.size() > common_prefix) {
            llama_memory_t mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;
            if (mem && g_llama.memory_seq_rm) {
                const int n_keep = 4;
                llama_pos live_rm_start;
                if (!ctx->ssd_blocks.empty()) {
                    // Compacted live window: positions 0..n_keep-1 are sink tokens,
                    // position n_keep onward maps to absolute token [ssd_evicted_tokens, ...)
                    live_rm_start = (llama_pos)(common_prefix - ctx->ssd_evicted_tokens) + n_keep + live_restored_tokens(ctx);
                } else {
                    // No SSD eviction: positions are 1:1 with absolute token indices
                    live_rm_start = (llama_pos)common_prefix;
                }
                // Safety: only remove if the computed position is within current live range
                llama_pos cur_max = (g_llama.memory_seq_pos_max && g_llama.get_memory)
                    ? g_llama.memory_seq_pos_max(g_llama.get_memory(ctx->ctx), 0)
                    : -1;
                if (live_rm_start >= 0 && (cur_max < 0 || live_rm_start <= cur_max)) {
                    g_llama.memory_seq_rm(mem, 0, live_rm_start, -1);
                }
            }
            int diff = (int)(ctx->history_tokens.size() - common_prefix);
            ctx->active_tokens = std::max(0, ctx->active_tokens - diff);
            ctx->total_tokens = (int)common_prefix;
            ctx->active_restored_block = -1;
            ctx->history_tokens.resize(common_prefix);
            // NOTE: Do NOT clear ssd_blocks or quest -- archived KV state remains valid
        }
        processed = (int)common_prefix;
        fprintf(stderr, "[OPREL SSD] SSD-aware prefix reuse: %zu tokens cached (%d SSD blocks + %d live tokens), decoding only %d delta tokens.\n",
                common_prefix, (int)ctx->ssd_blocks.size(), ctx->active_tokens, n_tokens - processed);
    }

    // Autonomous SSD Restorer Hook: If context has archived blocks, check query against Quest
    if (ctx->enable_autonomous_restore && ctx->ssd_blocks.size() > 0 && n_tokens > processed) {
        int query_len = n_tokens - processed;
        std::vector<float> q_summary(128, 0.0f);
        compute_token_summary(tokens.data() + processed, query_len, q_summary.data(), 128);

        int top_k = std::max(1, std::min((int)ctx->ssd_blocks.size(), ctx->retrieval_top_k));
        std::vector<int32_t> best_blks(top_k);
        int n_sel = quest_select_topk(ctx->quest, q_summary.data(), top_k, best_blks.data());
        if (n_sel > 0) {
            std::set<int> new_set;
            for (int i = 0; i < n_sel; i++) {
                if (best_blks[i] >= 0 && best_blks[i] < (int)ctx->ssd_blocks.size()) {
                    new_set.insert(best_blks[i]);
                }
            }

            llama_memory_t mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;

            // 1. Drop restored blocks that are no longer selected (closes the gap, re-tracks positions).
            std::vector<int> stale;
            for (int b : ctx->active_restored_blocks) {
                if (new_set.find(b) == new_set.end()) stale.push_back(b);
            }
            std::sort(stale.begin(), stale.end(), [&](int a, int b) {
                return ctx->ssd_blocks[a].live_pos_start > ctx->ssd_blocks[b].live_pos_start;
            });
            for (int b : stale) drop_restored_block(ctx, b);

            // 2. Restore selected blocks that are not live yet, in original token order. The ingest
            // decode below applies the lazy K-shift, so no extra refresh decode is needed here.
            for (int b : new_set) {
                bool already_active = false;
                for (int active_b : ctx->active_restored_blocks) {
                    if (active_b == b) { already_active = true; break; }
                }
                if (!already_active) {
                    fprintf(stderr, "[OPREL SSD] Multi-Block: Restoring block %d [%d, %d) into live attention...\n",
                            b, ctx->ssd_blocks[b].start_token,
                            ctx->ssd_blocks[b].start_token + ctx->ssd_blocks[b].n_tokens);
                    restore_block(ctx, b, false);
                }
            }
            ctx->active_restored_block = ctx->active_restored_blocks.empty() ? -1 : ctx->active_restored_blocks.back();
        }
    }

    // Ingest remaining tokens in parallel chunks of 512 for optimal GPU prefill without memory spikes
    const int CHUNK_SIZE = 512;

    while (processed < n_tokens) {
        int count = std::min(CHUNK_SIZE, n_tokens - processed);

        // When nearing active context limit, snapshot chunk to SSD and shift KV memory slots
        // Crucial invariant: ALWAYS preserve a live window of recent tokens so sequence 0's max pos == ctx->total_tokens - 1
        const int min_live_window = std::min(256, std::max(64, (ctx->active_ctx_size - 256) / 2));
        const int n_keep = 4;
        if (ctx->active_tokens + count >= ctx->active_ctx_size - 128 && ctx->active_tokens > n_keep + min_live_window) {
            int max_discardable = ctx->active_tokens - n_keep - min_live_window;
            int n_discard = std::min(max_discardable, 1024);
            if (n_discard >= 128) {
                archive_and_evict_kv(ctx, n_discard, n_keep);
            }
        }

        for (int i = 0; i < count; i++) {
            ctx->history_tokens.push_back(tokens[processed + i]);
        }

        bool is_last_chunk = (processed + count == n_tokens);
        llama_batch batch;
        bool used_batch_init = false;

        llama_memory_t cur_mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;
        llama_pos cur_start_pos = (cur_mem && g_llama.memory_seq_pos_max) ? g_llama.memory_seq_pos_max(cur_mem, 0) + 1 : 0;
        if (cur_start_pos < 0) cur_start_pos = 0;

        if (g_llama.batch_init) {
            batch = g_llama.batch_init(count, 0, 1);
            batch.n_tokens = count;
            used_batch_init = true;
            for (int i = 0; i < count; i++) {
                batch.token[i] = tokens[processed + i];
                batch.pos[i] = cur_start_pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                // Only request logits for the absolute last token of the whole prompt (n_outputs = 1)
                // Intermediate tokens have logits = 0 (n_outputs = 0), reducing prefill latency drastically
                batch.logits[i] = (is_last_chunk && i == count - 1) ? 1 : 0;
            }
        } else {
            batch = g_llama.batch_get_one ? g_llama.batch_get_one(tokens.data() + processed, count) : llama_batch{0};
            if (batch.pos) {
                for (int i = 0; i < count; i++) {
                    batch.pos[i] = cur_start_pos + i;
                }
            }
        }

        int res = g_llama.decode(ctx->ctx, batch);
        if (res == 0) {
            ctx->stats.decode_calls++;
            ctx->shift_pending = false;  // llama_decode applied any pending K-shift
        }
        if (used_batch_init && g_llama.batch_free) {
            g_llama.batch_free(batch);
        }

        if (res != 0) {
            fprintf(stderr, "[OPREL C++] llama_decode error %d at pos %d\n", res, ctx->total_tokens);
            break;
        }

        ctx->total_tokens += count;
        ctx->active_tokens += count;
        processed += count;
    }

    ctx->stats.total_tokens = ctx->total_tokens;
    ctx->stats.active_tokens = ctx->active_tokens;
    ctx->stats.ssd_blocks = (int)ctx->ssd_blocks.size();

    return n_tokens;
}

OPREL_API int oprel_ingest_prompt(
    OprelContext* ctx,
    const char* prompt,
    bool reset_context
) {
#ifdef _WIN32
    __try {
        return internal_ingest_prompt(ctx, prompt, reset_context);
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        unsigned long code = GetExceptionCode();
        fprintf(stderr, "[OPREL C++ FATAL] oprel_ingest_prompt crashed with exception code: 0x%08X\n", code);
        return -1;
    }
#else
    return internal_ingest_prompt(ctx, prompt, reset_context);
#endif
}

OPREL_API int oprel_generate_next_token_ex(
    OprelContext* ctx,
    float temperature,
    float top_p,
    float repeat_penalty,
    float freq_penalty,
    float presence_penalty
) {
    if (!ctx) return -1;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    auto t0 = std::chrono::high_resolution_clock::now();

    llama_memory_t mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;
    llama_pos cur_max_pos = (mem && g_llama.memory_seq_pos_max) ? g_llama.memory_seq_pos_max(mem, 0) : (ctx->active_tokens - 1);

    // 1. Sliding window management: ensure room in active KV window before decoding
    const int min_live_window = 256;
    const int n_keep = 4;
    if (cur_max_pos >= ctx->active_ctx_size - 64 && ctx->active_tokens > n_keep + min_live_window) {
        int max_discardable = ctx->active_tokens - n_keep - min_live_window;
        int n_discard = std::min(max_discardable, 512);
        if (n_discard >= 64) {
            archive_and_evict_kv(ctx, n_discard, n_keep);
            cur_max_pos = (mem && g_llama.memory_seq_pos_max) ? g_llama.memory_seq_pos_max(mem, 0) : (ctx->active_tokens - 1);
        }
    }

    // Safety guard: gracefully stop only if maximum SSD capacity limit (131,072 tokens) reached
    if (ctx->total_tokens >= ctx->max_ssd_context) {
        fprintf(stderr, "[OPREL C++] Max SSD context capacity reached (%d tokens), stopping cleanly.\n",
                ctx->max_ssd_context);
        const llama_vocab* vocab = ctx->model_ref ? ctx->model_ref->vocab : NULL;
        llama_token eos_id = (vocab && g_llama.vocab_eos) ? g_llama.vocab_eos(vocab) : 1;
        return (int)eos_id;
    }

    // 2. Sample next token with penalties, top-k, top-p, temp (cached chain for maximum tokens/sec)
    if (!ctx->sampler_chain || 
        ctx->last_temperature != temperature || 
        ctx->last_top_p != top_p ||
        ctx->last_repeat_penalty != repeat_penalty ||
        ctx->last_freq_penalty != freq_penalty ||
        ctx->last_presence_penalty != presence_penalty) {

        if (ctx->sampler_chain && g_llama.sampler_free) {
            g_llama.sampler_free(ctx->sampler_chain);
            ctx->sampler_chain = NULL;
        }

        llama_sampler_chain_params sparams = g_llama.sampler_chain_default_params ? g_llama.sampler_chain_default_params() : llama_sampler_chain_params{false};
        ctx->sampler_chain = g_llama.sampler_chain_init ? g_llama.sampler_chain_init(sparams) : NULL;

        if (ctx->sampler_chain) {
            const llama_vocab* vocab = ctx->model_ref ? ctx->model_ref->vocab : NULL;
            int n_vocab = (vocab && g_llama.vocab_n_tokens) ? g_llama.vocab_n_tokens(vocab) : 262208;

            float r_pen = repeat_penalty > 0.0f ? repeat_penalty : 1.15f;
            float f_pen = freq_penalty >= 0.0f ? freq_penalty : 0.35f;
            float p_pen = presence_penalty >= 0.0f ? presence_penalty : 0.30f;

            if (g_llama.sampler_init_penalties) {
                g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_penalties(
                    256,    // penalty_last_n: 256 tokens
                    r_pen,  // penalty_repeat: 1.15
                    f_pen,  // penalty_freq: 0.35
                    p_pen   // penalty_present: 0.30
                ));
            }

            if (g_llama.sampler_init_dry && vocab) {
                g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_dry(
                    vocab,
                    0.8f,   // dry_multiplier
                    1.75f,  // dry_base
                    2,      // dry_allowed_length
                    256,    // dry_penalty_last_n
                    NULL,   // seq_breakers
                    0       // num_breakers
                ));
            }

            if (temperature <= 0.0f) {
                if (g_llama.sampler_init_greedy) {
                    g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_greedy());
                }
            } else {
                if (g_llama.sampler_init_top_k) {
                    g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_top_k(40));
                }
                if (top_p < 1.0f && g_llama.sampler_init_top_p) {
                    g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_top_p(top_p, 1));
                }
                if (g_llama.sampler_init_temp) {
                    g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_temp(temperature));
                }
                if (g_llama.sampler_init_dist) {
                    g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_dist(42));
                } else if (g_llama.sampler_init_greedy) {
                    g_llama.sampler_chain_add(ctx->sampler_chain, g_llama.sampler_init_greedy());
                }
            }
            ctx->last_temperature = temperature;
            ctx->last_top_p = top_p;
            ctx->last_repeat_penalty = repeat_penalty;
            ctx->last_freq_penalty = freq_penalty;
            ctx->last_presence_penalty = presence_penalty;

            // Seed recent history into penalty and DRY samplers so they immediately penalize repetition
            if (ctx->sampler_chain && g_llama.sampler_accept && !ctx->history_tokens.empty()) {
                int seed_count = std::min((int)ctx->history_tokens.size(), 256);
                size_t start_idx = ctx->history_tokens.size() - seed_count;
                for (size_t i = start_idx; i < ctx->history_tokens.size(); i++) {
                    g_llama.sampler_accept(ctx->sampler_chain, ctx->history_tokens[i]);
                }
            }
        }
    }

    llama_token token = -1;
    if (ctx->sampler_chain && g_llama.sampler_sample) {
        token = g_llama.sampler_sample(ctx->sampler_chain, ctx->ctx, -1);
        if (token >= 0 && g_llama.sampler_accept) {
            g_llama.sampler_accept(ctx->sampler_chain, token);
        }
    }

    // 3. Decode sampled token to advance context position using preallocated single batch
    if (token >= 0) {
        cur_max_pos = (mem && g_llama.memory_seq_pos_max) ? g_llama.memory_seq_pos_max(mem, 0) : (ctx->active_tokens - 1);
        ctx->single_token_val = token;
        ctx->single_pos_val = (cur_max_pos >= 0) ? (cur_max_pos + 1) : 0;
        int d_res = g_llama.decode(ctx->ctx, ctx->single_batch);
        if (d_res != 0) {
            fprintf(stderr, "[OPREL C++] llama_decode single token error %d at pos %d\n", d_res, (int)ctx->single_pos_val);
            return -1;
        }
        ctx->stats.decode_calls++;
        ctx->shift_pending = false;

        ctx->history_tokens.push_back(token);
        ctx->total_tokens++;
        ctx->active_tokens++;
        ctx->stats.total_tokens = ctx->total_tokens;
        ctx->stats.active_tokens = ctx->active_tokens;
        ctx->stats.ssd_blocks = (int)ctx->ssd_blocks.size();
    }

    return (int)token;
}

OPREL_API int oprel_generate_next_token(
    OprelContext* ctx,
    float temperature,
    float top_p
) {
    return oprel_generate_next_token_ex(ctx, temperature, top_p, 1.15f, 0.35f, 0.30f);
}

OPREL_API int oprel_retrieve_context(
    OprelContext* ctx,
    const char* query,
    int top_k,
    char* out_buf,
    int max_len
) {
    if (!ctx || !query || !out_buf || max_len <= 0) return 0;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    out_buf[0] = '\0';
    if (ctx->ssd_blocks.empty() || !ctx->quest) return 0;

    const llama_vocab* vocab = ctx->model_ref->vocab;
    int n_q_chars = (int)strlen(query);
    std::vector<llama_token> q_tokens(n_q_chars + 32);
    int n_tokens = g_llama.tokenize ? g_llama.tokenize(vocab, query, n_q_chars, q_tokens.data(), (int)q_tokens.size(), false, true) : 0;
    if (n_tokens <= 0) return 0;

    // Compute query summary vector
    std::vector<float> q_summary(128, 0.0f);
    compute_token_summary(q_tokens.data(), n_tokens, q_summary.data(), 128);

    // Query Quest index for top-K matching block indices
    std::vector<int32_t> selected_blocks(top_k > 0 ? top_k : 1);
    int n_selected = quest_select_topk(ctx->quest, q_summary.data(), (int32_t)selected_blocks.size(), selected_blocks.data());
    if (n_selected <= 0) return 0;

    int total_chars_written = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    // Pure KV architecture: retrieval returns block IDs only.
    // Text reconstruction from history_tokens is disabled and unreachable.
    out_buf[0] = '\0';
    return 0;
}

static bool internal_load_and_inject_block(OprelContext* ctx, int block_index, bool refresh_logits) {
    if (!ctx || block_index < 0 || block_index >= (int)ctx->ssd_blocks.size()) return false;
    if (!ctx->model_ref || !g_llama.get_memory || !g_llama.memory_seq_cp ||
        !g_llama.memory_seq_rm || !g_llama.memory_seq_add ||
        !g_llama.state_seq_set_data_ext) return false;

    const SSDBlockRecord& rec = ctx->ssd_blocks[block_index];
    if (rec.start_token < 0 || rec.n_tokens <= 0 || rec.file_offset < 0) {
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    int64_t decode_before = ctx->stats.decode_calls;
    join_bg_write(ctx);
    // join_bg_write may have just marked this very block unrestorable.
    if (rec.file_offset < 0) return false;

    size_t total_payload = sizeof(OprelSSDBlockHeader) + rec.state_bytes;
    size_t aligned_sz = (total_payload + OPREL_SECTOR_SIZE - 1) & ~(size_t)(OPREL_SECTOR_SIZE - 1);
    if (rec.state_bytes == 0 || aligned_sz < sizeof(OprelSSDBlockHeader) ||
        (uint64_t)rec.file_offset + aligned_sz > (uint64_t)ctx->ssd_file.current_size) {
        ctx->stats.archive_validation_failures++;
        return false;
    }
    if (!ensure_io_buffer(ctx, aligned_sz) ||
        !direct_io_read(&ctx->ssd_file, rec.file_offset, ctx->io_aligned_buffer, aligned_sz)) {
        ctx->stats.archive_validation_failures++;
        return false;
    }
    auto read_done = std::chrono::high_resolution_clock::now();
    ctx->stats.total_ssd_bytes_read += (int64_t)aligned_sz;
    ctx->stats.last_block_bytes = (int64_t)rec.state_bytes;

    const OprelSSDBlockHeader* hdr = (const OprelSSDBlockHeader*)ctx->io_aligned_buffer;
    bool valid = memcmp(hdr->magic, "OPREL_KV", 8) == 0 &&
        hdr->format_version == OPREL_BLOCK_FORMAT_VERSION &&
        hdr->llama_build == OPREL_LLAMA_BUILD &&
        memcmp(hdr->model_sha256, ctx->model_ref->artifact_sha256, 32) == 0 &&
        g_runtime_sha256_valid &&
        memcmp(hdr->runtime_sha256, g_runtime_sha256, 32) == 0 &&
        hdr->n_layer == ctx->model_ref->n_layer &&
        hdr->type_k == ctx->kv_type && hdr->type_v == ctx->kv_type &&
        hdr->p0 >= 0 && hdr->p1 > hdr->p0 && hdr->n_tokens == hdr->p1 - hdr->p0 &&
        hdr->n_tokens == rec.n_tokens && hdr->p0 == rec.start_token &&
        hdr->state_bytes == rec.state_bytes &&
        hdr->state_bytes <= aligned_sz - sizeof(OprelSSDBlockHeader) &&
        calculate_crc32((const uint8_t*)ctx->io_aligned_buffer + sizeof(OprelSSDBlockHeader), hdr->state_bytes) == hdr->checksum;
    if (!valid) {
        ctx->stats.archive_validation_failures++;
        fprintf(stderr, "[OPREL SSD] Rejected incompatible or corrupted archive block %d.\n", block_index);
        return false;
    }
    ctx->stats.archive_validation_passes++;
    auto validation_done = std::chrono::high_resolution_clock::now();

    llama_memory_t mem = g_llama.get_memory(ctx->ctx);
    if (!mem) return false;
    const int cs = (int)hdr->reserved;   // live position the cells were serialised at
    const int n = hdr->n_tokens;
    llama_pos live_max = g_llama.memory_seq_pos_max(mem, 0);

    // Insert right after the sinks, behind already-restored blocks that precede this one in the
    // original token order. The live tail shifts right by n, which for the most recent evicted
    // block puts every token back at its original position.
    int target = OPREL_N_KEEP;
    for (int b : ctx->active_restored_blocks) {
        if (b < 0 || b >= (int)ctx->ssd_blocks.size()) continue;
        const SSDBlockRecord& r = ctx->ssd_blocks[b];
        if (r.live_pos_start >= 0 && r.start_token < rec.start_token) {
            target = std::max(target, r.live_pos_start + r.n_tokens);
        }
    }
    // The tail shift only keeps the sequence contiguous if the live cells reach the insertion
    // point; otherwise reject up front instead of mutating and rolling back.
    if (cs < 0 || live_max < target - 1) {
        fprintf(stderr, "[OPREL SSD] Block %d cannot be inserted at %d: live max pos is %d.\n", block_index, target, (int)live_max);
        return false;
    }

    g_llama.memory_seq_rm(mem, 1, -1, -1);
    const uint8_t* payload = (const uint8_t*)ctx->io_aligned_buffer + sizeof(OprelSSDBlockHeader);
    size_t restored_sz = g_llama.state_seq_set_data_ext(ctx->ctx, payload, hdr->state_bytes, 1, 0);
    if (restored_sz != hdr->state_bytes) {
        g_llama.memory_seq_rm(mem, 1, -1, -1);
        return false;
    }
    if (g_llama.memory_seq_pos_min(mem, 1) != cs || g_llama.memory_seq_pos_max(mem, 1) != cs + n - 1) {
        g_llama.memory_seq_rm(mem, 1, -1, -1);
        return false;
    }
    auto set_done = std::chrono::high_resolution_clock::now();

    // Re-position the restored cells and open a gap in the live window. Both are lazy K-shifts that
    // the next llama_decode applies (a pending shift is tracked in ctx->shift_pending).
    if (target != cs) g_llama.memory_seq_add(mem, 1, cs, cs + n, target - cs);
    g_llama.memory_seq_add(mem, 0, target, -1, n);
    g_llama.memory_seq_cp(mem, 1, 0, target, target + n);
    if (g_llama.memory_seq_pos_max(mem, 0) != live_max + n) {
        // Drop the cells just copied first, then close the gap; the other order would drag the
        // copied cells to the wrong positions and leave them live.
        g_llama.memory_seq_rm(mem, 0, target, target + n);
        g_llama.memory_seq_add(mem, 0, target, -1, -n);
        g_llama.memory_seq_rm(mem, 1, -1, -1);
        return false;
    }
    g_llama.memory_seq_rm(mem, 1, -1, -1);
    ctx->shift_pending = true;

    // Blocks already restored at or after the insertion point moved right with the tail.
    for (SSDBlockRecord& o : ctx->ssd_blocks) {
        if (o.live_pos_start >= target) o.live_pos_start += n;
    }
    ctx->ssd_blocks[block_index].live_pos_start = target;
    ctx->active_tokens += n;

    // The sampler must see the restored prefix, not logits computed without it.
    auto tr0 = std::chrono::high_resolution_clock::now();
    if (refresh_logits && !apply_pending_shift(ctx)) {
        fprintf(stderr, "[OPREL SSD] Block %d restored, but the logits refresh failed.\n", block_index);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double refresh_ms = std::chrono::duration<double, std::milli>(t1 - tr0).count();
    ctx->stats.last_read_ms = std::chrono::duration<double, std::milli>(read_done - t0).count();
    ctx->stats.last_set_data_ms = std::chrono::duration<double, std::milli>(set_done - validation_done).count();
    ctx->stats.last_restore_latency_ms = ms;
    // Counts the single refresh decode of the last live token (if requested); the restored
    // block's own tokens are never decoded, which restore_prefill_tokens records.
    ctx->stats.restore_decode_calls = ctx->stats.decode_calls - decode_before;
    ctx->stats.restore_prefill_tokens = 0;
    ctx->stats.last_restore_pos_start = target;
    ctx->stats.last_restore_pos_end = target + n;
    ctx->stats.last_restore_kv_match = -1; // set by explicit reference comparisons in the tests
    ctx->stats.retrieved_blocks++;
    ctx->stats.last_retrieval_ms = ms;

    fprintf(stderr, "[OPREL SSD] Restored Block %d from SSD (%d tokens) in %.2f ms | stored_pos=[%d, %d) live_pos=[%d, %d) | decode_calls=%lld prefill_tokens=0 | read %.1f ms, set_data %.1f ms, refresh(K-shift+1 decode) %.1f ms\n",
            block_index, n, ms, cs, cs + n, target, target + n, (long long)ctx->stats.restore_decode_calls,
            ctx->stats.last_read_ms, ctx->stats.last_set_data_ms, refresh_ms);

    return true;
}

static bool restore_block(OprelContext* ctx, int block_index, bool refresh_logits) {
    if (!ctx || block_index < 0 || block_index >= (int)ctx->ssd_blocks.size()) return false;
    for (int b : ctx->active_restored_blocks) {
        if (b == block_index) return true;
    }
    bool ok = internal_load_and_inject_block(ctx, block_index, refresh_logits);
    if (ok) {
        ctx->active_restored_blocks.push_back(block_index);
        ctx->active_restored_block = block_index;
        sync_stats(ctx);
    }
    return ok;
}

OPREL_API bool oprel_restore_kv_state(
    OprelContext* ctx,
    int block_index
) {
    if (!ctx) return false;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    return restore_block(ctx, block_index, true);
}

OPREL_API bool oprel_refresh_logits(OprelContext* ctx) {
    if (!ctx) return false;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    // Re-decodes the last live token against the current cache (single-token decode, no prefix).
    ctx->shift_pending = true;
    return apply_pending_shift(ctx);
}

OPREL_API bool oprel_restore_kv_block(OprelContext* ctx, int block_index, bool refresh_logits) {
    if (!ctx) return false;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    return restore_block(ctx, block_index, refresh_logits);
}

OPREL_API int oprel_archive_oldest(OprelContext* ctx, int n_tokens) {
    if (!ctx || n_tokens <= 0) return -1;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    // Keep at least the sinks plus one live token.
    if (n_tokens > ctx->active_tokens - OPREL_N_KEEP - 1) return -1;
    if (!archive_and_evict_kv(ctx, n_tokens, OPREL_N_KEEP)) return -1;
    sync_stats(ctx);
    return (int)ctx->ssd_blocks.size() - 1;
}

OPREL_API int oprel_find_best_block(
    OprelContext* ctx,
    const char* query
) {
    if (!ctx || !query || ctx->ssd_blocks.empty() || !ctx->quest) return -1;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);

    const llama_vocab* vocab = ctx->model_ref->vocab;
    int n_q_chars = (int)strlen(query);
    std::vector<llama_token> q_tokens(n_q_chars + 32);
    int n_tokens = g_llama.tokenize ? g_llama.tokenize(vocab, query, n_q_chars, q_tokens.data(), (int)q_tokens.size(), false, true) : 0;
    if (n_tokens <= 0) return -1;

    std::vector<float> q_summary(128, 0.0f);
    compute_token_summary(q_tokens.data(), n_tokens, q_summary.data(), 128);

    std::vector<int32_t> selected_blocks(1);
    int n_selected = quest_select_topk(ctx->quest, q_summary.data(), 1, selected_blocks.data());
    if (n_selected <= 0) return -1;
    return selected_blocks[0];
}

OPREL_API int oprel_score_all_blocks(
    OprelContext* ctx,
    const char* query,
    int32_t* out_block_indices,
    float* out_scores,
    int32_t* out_start_tokens,
    int32_t* out_n_tokens,
    int max_blocks
) {
    if (!ctx || !query || ctx->ssd_blocks.empty() || !ctx->quest || max_blocks <= 0) return 0;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);

    const llama_vocab* vocab = ctx->model_ref->vocab;
    int n_q_chars = (int)strlen(query);
    std::vector<llama_token> q_tokens(n_q_chars + 32);
    int n_tokens = g_llama.tokenize ? g_llama.tokenize(vocab, query, n_q_chars, q_tokens.data(), (int)q_tokens.size(), false, true) : 0;
    if (n_tokens <= 0) return 0;

    std::vector<float> q_summary(128, 0.0f);
    compute_token_summary(q_tokens.data(), n_tokens, q_summary.data(), 128);

    return quest_get_all_scores(ctx->quest, q_summary.data(), out_block_indices, out_scores, out_start_tokens, out_n_tokens, max_blocks);
}

OPREL_API int oprel_token_to_text(
    OprelContext* ctx,
    int token,
    char* out_buf,
    int max_len
) {
    if (!ctx || !out_buf || max_len <= 0) return 0;
    const llama_vocab* vocab = ctx->model_ref->vocab;
    if (!g_llama.token_to_piece) return 0;
    return g_llama.token_to_piece(vocab, (llama_token)token, out_buf, max_len, 0, false);
}

OPREL_API bool oprel_is_eog(OprelContext* ctx, int token) {
    if (!ctx) return true;
    const llama_vocab* vocab = ctx->model_ref->vocab;
    if (!g_llama.token_is_eog) return false;
    return g_llama.token_is_eog(vocab, (llama_token)token);
}

OPREL_API void oprel_get_stats(OprelContext* ctx, OprelStats* out_stats) {
    if (!ctx || !out_stats) return;
    *out_stats = ctx->stats;
}

OPREL_API int oprel_get_logits(OprelContext* ctx, float* out_logits, int max_vocab) {
    if (!ctx || !out_logits || max_vocab <= 0 || !g_llama.get_logits_ith) return 0;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    float* l = g_llama.get_logits_ith(ctx->ctx, -1);
    if (!l) return 0;
    int n_vocab = g_llama.vocab_n_tokens ? g_llama.vocab_n_tokens(ctx->model_ref->vocab) : 262208;
    int count = std::min(n_vocab, max_vocab);
    memcpy(out_logits, l, count * sizeof(float));
    return count;
}

OPREL_API int oprel_reserialize_range(OprelContext* ctx, int p0, int p1, uint8_t* out_buf, int max_len) {
    if (!ctx || p0 < 0 || p1 <= p0) return 0;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    if (!g_llama.get_memory || !g_llama.memory_seq_cp || !g_llama.memory_seq_rm) return 0;
    if (!g_llama.state_seq_get_size_ext || !g_llama.state_seq_get_data_ext) return 0;

    if (!apply_pending_shift(ctx)) return 0;
    llama_memory_t mem = g_llama.get_memory(ctx->ctx);
    g_llama.memory_seq_rm(mem, 1, -1, -1);
    g_llama.memory_seq_cp(mem, 0, 1, p0, p1);
    size_t sz = g_llama.state_seq_get_size_ext(ctx->ctx, 1, 0);
    if (sz > 0 && out_buf && max_len >= (int)sz) {
        g_llama.state_seq_get_data_ext(ctx->ctx, out_buf, sz, 1, 0);
    }
    g_llama.memory_seq_rm(mem, 1, -1, -1);
    return (int)sz;
}

OPREL_API int oprel_get_block_payload(OprelContext* ctx, int block_index, uint8_t* out_buf, int max_len) {
    if (!ctx || block_index < 0 || block_index >= (int)ctx->ssd_blocks.size()) return 0;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    join_bg_write(ctx);
    const SSDBlockRecord& rec = ctx->ssd_blocks[block_index];
    if (rec.file_offset < 0 || rec.state_bytes == 0) return 0;
    size_t total_payload = sizeof(OprelSSDBlockHeader) + rec.state_bytes;
    size_t aligned_sz = (total_payload + OPREL_SECTOR_SIZE - 1) & ~(size_t)(OPREL_SECTOR_SIZE - 1);
    if ((uint64_t)rec.file_offset + aligned_sz > (uint64_t)ctx->ssd_file.current_size) return 0;

    if (!ensure_io_buffer(ctx, aligned_sz)) return 0;

    bool ok = direct_io_read(&ctx->ssd_file, rec.file_offset, ctx->io_aligned_buffer, aligned_sz);
    if (!ok) return 0;

    const OprelSSDBlockHeader* hdr = (const OprelSSDBlockHeader*)ctx->io_aligned_buffer;
    if (memcmp(hdr->magic, "OPREL_KV", 8) != 0) return 0;
    if (hdr->state_bytes != rec.state_bytes ||
        calculate_crc32((const uint8_t*)ctx->io_aligned_buffer + sizeof(OprelSSDBlockHeader), hdr->state_bytes) != hdr->checksum) {
        ctx->stats.archive_validation_failures++;
        return 0;
    }

    const uint8_t* payload = (const uint8_t*)ctx->io_aligned_buffer + sizeof(OprelSSDBlockHeader);
    int to_copy = std::min((int)hdr->state_bytes, max_len);
    if (out_buf && to_copy > 0) {
        memcpy(out_buf, payload, to_copy);
    }
    return (int)hdr->state_bytes;
}

// OprelStats mirrors the counters Python reads; refresh them whenever the context state is replaced.
static void sync_stats(OprelContext* ctx) {
    ctx->stats.total_tokens = ctx->total_tokens;
    ctx->stats.active_tokens = ctx->active_tokens;
    ctx->stats.ssd_blocks = (int)ctx->ssd_blocks.size();
}

static void buf_append(std::vector<uint8_t>& buf, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    buf.insert(buf.end(), p, p + len);
}

template <typename T>
static void buf_put(std::vector<uint8_t>& buf, T v) { buf_append(buf, &v, sizeof(T)); }

// Bounds-checked sequential reader; any over-read latches `ok = false` and returns zeros.
struct ByteReader {
    const uint8_t* p;
    size_t n;
    size_t off = 0;
    bool ok = true;
    ByteReader(const uint8_t* data, size_t size) : p(data), n(size) {}
    bool read(void* dst, size_t len) {
        if (!ok || len > n - off) { ok = false; return false; }
        if (len) memcpy(dst, p + off, len);
        off += len;
        return true;
    }
    template <typename T> T get() { T v{}; read(&v, sizeof(T)); return v; }
    size_t remaining() const { return ok ? n - off : 0; }
};

// Session file v5 layout (all little-endian, trailing CRC32 over everything before it):
//   u32 magic 'OPSN' | u32 version | u8[32] model sha256 | i32 type_k | i32 type_v | i32 n_layer
//   i32 total_tokens | i32 active_tokens | i32 active_restored_block
//   u32 n_active_restored | i32[n]            (blocks whose cells are live in the saved KV)
//   u32 n_hist | llama_token[n]
//   u32 record_size | u32 n_blocks | SSDBlockRecord[n]
//   u64 kv_bytes | kv payload
//   u32 crc32
// The file is written to <path>.tmp, fsynced, then atomically renamed over <path>.
static bool internal_save_session(OprelContext* ctx, const std::string& path) {
    if (!ctx || path.empty() || !g_llama.state_seq_get_size_ext || !g_llama.state_seq_get_data_ext) return false;

    join_bg_write(ctx);
    // A pending K-shift would be serialised un-rotated; apply it first.
    if (!apply_pending_shift(ctx)) return false;
    // Make the block file durable before a session file that references it becomes visible.
    if (ctx->ssd_file.is_open) {
#ifdef _WIN32
        FlushFileBuffers(ctx->ssd_file.handle);
#else
        fsync(ctx->ssd_file.fd);
#endif
    }

    size_t live_sz = g_llama.state_seq_get_size_ext(ctx->ctx, 0, 0);
    std::vector<uint8_t> kv(live_sz);
    if (live_sz > 0 && g_llama.state_seq_get_data_ext(ctx->ctx, kv.data(), live_sz, 0, 0) != live_sz) {
        fprintf(stderr, "[OPREL SSD] Session save aborted: live KV serialisation was short.\n");
        return false;
    }

    std::vector<uint8_t> buf;
    buf.reserve(256 + ctx->history_tokens.size() * sizeof(llama_token) +
                ctx->ssd_blocks.size() * sizeof(SSDBlockRecord) + live_sz);
    buf_put<uint32_t>(buf, 0x4E53504F);  // 'OPSN'
    buf_put<uint32_t>(buf, OPREL_SESSION_VERSION);
    buf_append(buf, ctx->model_ref->artifact_sha256, 32);
    buf_put<int32_t>(buf, ctx->kv_type);
    buf_put<int32_t>(buf, ctx->kv_type);
    buf_put<int32_t>(buf, ctx->model_ref->n_layer);
    buf_put<int32_t>(buf, ctx->total_tokens);
    buf_put<int32_t>(buf, ctx->active_tokens);
    buf_put<int32_t>(buf, ctx->active_restored_block);
    buf_put<uint32_t>(buf, (uint32_t)ctx->active_restored_blocks.size());
    for (int b : ctx->active_restored_blocks) buf_put<int32_t>(buf, b);
    buf_put<uint32_t>(buf, (uint32_t)ctx->history_tokens.size());
    if (!ctx->history_tokens.empty()) {
        buf_append(buf, ctx->history_tokens.data(), ctx->history_tokens.size() * sizeof(llama_token));
    }
    buf_put<uint32_t>(buf, (uint32_t)sizeof(SSDBlockRecord));
    buf_put<uint32_t>(buf, (uint32_t)ctx->ssd_blocks.size());
    if (!ctx->ssd_blocks.empty()) {
        buf_append(buf, ctx->ssd_blocks.data(), ctx->ssd_blocks.size() * sizeof(SSDBlockRecord));
    }
    buf_put<uint64_t>(buf, (uint64_t)live_sz);
    if (live_sz > 0) buf_append(buf, kv.data(), live_sz);
    buf_put<uint32_t>(buf, calculate_crc32(buf.data(), buf.size()));

    const std::string tmp = path + ".tmp";
    FILE* fp = fopen(tmp.c_str(), "wb");
    if (!fp) return false;
    bool ok = fwrite(buf.data(), 1, buf.size(), fp) == buf.size() && flush_to_disk(fp);
    ok = (fclose(fp) == 0) && ok;
    if (!ok || !atomic_replace(tmp, path)) {
        fprintf(stderr, "[OPREL SSD] Session save failed for %s; previous session file (if any) left intact.\n", path.c_str());
        remove(tmp.c_str());
        return false;
    }
    return true;
}

static void reset_ctx_state(OprelContext* ctx) {
    llama_memory_t mem = g_llama.get_memory ? g_llama.get_memory(ctx->ctx) : NULL;
    if (mem && g_llama.memory_clear) g_llama.memory_clear(mem, true);
    ctx->total_tokens = 0;
    ctx->active_tokens = 0;
    ctx->ssd_evicted_tokens = 0;
    ctx->active_restored_block = -1;
    ctx->active_restored_blocks.clear();
    ctx->history_tokens.clear();
    ctx->ssd_blocks.clear();
    ctx->shift_pending = false;
    sync_stats(ctx);
}

// Loads and validates the whole session file in memory first; the context is only modified once
// every check passed. A corrupt, truncated or incompatible file is deleted and reported as a miss.
static bool internal_load_session(OprelContext* ctx, const std::string& path) {
    if (!ctx || path.empty() || !g_llama.state_seq_set_data_ext || !g_llama.get_memory || !g_llama.memory_clear) return false;

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::vector<uint8_t> file;
    {
        uint8_t chunk[1 << 16];
        size_t n;
        while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) file.insert(file.end(), chunk, chunk + n);
    }
    bool read_err = ferror(fp) != 0;
    fclose(fp);
    if (read_err) return false;  // transient I/O error: keep the file for a later attempt

    auto reject = [&](const char* why) {
        fprintf(stderr, "[OPREL SSD] Rejected session file %s: %s. Resetting session cleanly.\n", path.c_str(), why);
        remove(path.c_str());
        return false;
    };

    if (file.size() < sizeof(uint32_t) * 3) return reject("truncated");
    size_t body = file.size() - sizeof(uint32_t);
    uint32_t stored_crc;
    memcpy(&stored_crc, file.data() + body, sizeof(stored_crc));
    if (calculate_crc32(file.data(), body) != stored_crc) return reject("checksum mismatch (corrupt or truncated)");

    ByteReader r(file.data(), body);
    if (r.get<uint32_t>() != 0x4E53504F) return reject("bad magic");
    uint32_t version = r.get<uint32_t>();
    if (version != OPREL_SESSION_VERSION) return reject("incompatible session version");
    uint8_t sha[32] = {};
    r.read(sha, 32);
    if (!r.ok || memcmp(sha, ctx->model_ref->artifact_sha256, 32) != 0) return reject("model artifact mismatch");
    int32_t tk = r.get<int32_t>(), tv = r.get<int32_t>(), nl = r.get<int32_t>();
    if (!r.ok || tk != ctx->kv_type || tv != ctx->kv_type) return reject("incompatible KV cache type");
    if (nl != ctx->model_ref->n_layer) return reject("layer count mismatch");

    int32_t total_tokens = r.get<int32_t>();
    int32_t active_tokens = r.get<int32_t>();
    int32_t active_restored_block = r.get<int32_t>();
    uint32_t n_active = r.get<uint32_t>();
    if (!r.ok || total_tokens < 0 || active_tokens < 0 || n_active > r.remaining() / sizeof(int32_t)) return reject("bad counters");
    std::vector<int> active_blocks(n_active);
    for (uint32_t i = 0; i < n_active; i++) active_blocks[i] = r.get<int32_t>();

    uint32_t n_hist = r.get<uint32_t>();
    if (!r.ok || n_hist > r.remaining() / sizeof(llama_token)) return reject("bad history length");
    std::vector<llama_token> history(n_hist);
    if (n_hist > 0) r.read(history.data(), (size_t)n_hist * sizeof(llama_token));

    uint32_t rec_size = r.get<uint32_t>();
    uint32_t n_blks = r.get<uint32_t>();
    if (!r.ok || rec_size != sizeof(SSDBlockRecord) || n_blks > r.remaining() / sizeof(SSDBlockRecord)) return reject("bad block table");
    std::vector<SSDBlockRecord> blocks(n_blks);
    if (n_blks > 0) r.read(blocks.data(), (size_t)n_blks * sizeof(SSDBlockRecord));

    uint64_t kv_bytes = r.get<uint64_t>();
    if (!r.ok || kv_bytes != r.remaining()) return reject("bad KV payload length");
    const uint8_t* kv_ptr = file.data() + r.off;

    // Per-block sanity. A block that is out of range, or whose bytes are not inside the block file
    // (e.g. the .kv file was removed or truncated), stays in the table but becomes unrestorable.
    for (uint32_t i = 0; i < n_blks; i++) {
        SSDBlockRecord& b = blocks[i];
        if (b.block_index != (int32_t)i || b.start_token < 0 || b.n_tokens <= 0) return reject("inconsistent block record");
        if (b.file_offset >= 0) {
            size_t total = sizeof(OprelSSDBlockHeader) + b.state_bytes;
            size_t aligned = (total + OPREL_SECTOR_SIZE - 1) & ~(size_t)(OPREL_SECTOR_SIZE - 1);
            if (b.state_bytes == 0 || (uint64_t)b.file_offset + aligned > (uint64_t)ctx->ssd_file.current_size) {
                fprintf(stderr, "[OPREL SSD] Session block %u is not backed by the block file; marking unrestorable.\n", i);
                b.file_offset = -1;
            }
        }
    }
    for (int b : active_blocks) {
        if (b < 0 || b >= (int)n_blks) return reject("active block index out of range");
    }

    // Everything validated: now mutate the context.
    llama_memory_t mem = g_llama.get_memory(ctx->ctx);
    if (mem) g_llama.memory_clear(mem, true);
    ctx->shift_pending = false;
    if (kv_bytes > 0) {
        size_t restored_sz = g_llama.state_seq_set_data_ext(ctx->ctx, kv_ptr, (size_t)kv_bytes, 0, 0);
        if (restored_sz != kv_bytes) {
            fprintf(stderr, "[OPREL SSD] state_seq_set_data_ext restored %zu of %llu bytes during session load! Invalidating session.\n",
                    restored_sz, (unsigned long long)kv_bytes);
            remove(path.c_str());
            reset_ctx_state(ctx);
            return false;
        }
    }

    ctx->total_tokens = total_tokens;
    ctx->active_tokens = active_tokens;
    ctx->active_restored_block = active_restored_block;
    ctx->active_restored_blocks = active_blocks;
    ctx->history_tokens = std::move(history);
    ctx->ssd_blocks = std::move(blocks);
    // ssd_evicted_tokens is derived: the archive is a contiguous prefix of the history.
    ctx->ssd_evicted_tokens = 0;
    for (const SSDBlockRecord& rec : ctx->ssd_blocks) ctx->ssd_evicted_tokens += rec.n_tokens;

    if (ctx->quest) quest_free(ctx->quest);
    ctx->quest = quest_create(128, 512);
    for (const SSDBlockRecord& rec : ctx->ssd_blocks) {
        if (rec.file_offset < 0) continue;  // unrestorable blocks are never indexed
        std::vector<float> summary(128, 0.0f);
        if (rec.start_token >= 0 && (size_t)(rec.start_token + rec.n_tokens) <= ctx->history_tokens.size()) {
            compute_token_summary(&ctx->history_tokens[rec.start_token], rec.n_tokens, summary.data(), 128);
        }
        quest_add_block(ctx->quest, rec.block_index, rec.start_token, rec.n_tokens, summary.data(), summary.data(), summary.data());
    }
    sync_stats(ctx);
    return true;
}

static std::string session_dir_of(OprelContext* ctx) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path d = fs::path(ctx->ssd_cache_dir) / "sessions";
    fs::create_directories(d, ec);
    return d.string();
}

static std::string session_file_of(const std::string& dir, const std::string& sid) {
    return (std::filesystem::path(dir) / ("session_" + sid + ".bin")).string();
}

static std::string kv_file_of(OprelContext* ctx, const std::string& sid) {
    return (std::filesystem::path(ctx->ssd_cache_dir) / (sid + ".kv")).string();
}

// Sampler penalty/DRY state is seeded from the history; drop it when the history is replaced.
static void reset_sampler(OprelContext* ctx) {
    if (ctx->sampler_chain && g_llama.sampler_free) {
        g_llama.sampler_free(ctx->sampler_chain);
        ctx->sampler_chain = NULL;
    }
    ctx->last_temperature = -999.0f;
    ctx->last_top_p = -999.0f;
    ctx->last_repeat_penalty = -999.0f;
    ctx->last_freq_penalty = -999.0f;
    ctx->last_presence_penalty = -999.0f;
}

OPREL_API bool oprel_save_session(OprelContext* ctx) {
    if (!ctx) return false;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    if (ctx->current_session_id.empty() || ctx->current_session_id == "default" || ctx->total_tokens <= 0) {
        return false;
    }

    std::string path = session_file_of(session_dir_of(ctx), ctx->current_session_id);
    bool ok = internal_save_session(ctx, path);
    if (ok) {
        fprintf(stderr, "[OPREL SSD] Persisted session '%s' to SSD: %zu tokens (%d SSD blocks, %d active tokens).\n",
                ctx->current_session_id.c_str(), ctx->history_tokens.size(), (int)ctx->ssd_blocks.size(), ctx->active_tokens);
        prune_sessions(ctx);
    }
    return ok;
}

OPREL_API bool oprel_set_session(OprelContext* ctx, const char* session_id) {
    try {
    if (!ctx || !session_id || strlen(session_id) == 0) return false;
    std::lock_guard<std::recursive_mutex> lock(ctx->mu);
    const std::string new_sid = sanitize_session_id(session_id);
    if (ctx->current_session_id == new_sid) {
        return true; // Already active session
    }

    const std::string sdir = session_dir_of(ctx);

    // 1. Save the current session if active, and close its block file.
    join_bg_write(ctx);
    if (ctx->ssd_file.is_open) {
        if (!ctx->current_session_id.empty() && ctx->current_session_id != "default" && ctx->total_tokens > 0) {
            internal_save_session(ctx, session_file_of(sdir, ctx->current_session_id));
        }
        std::string prev_file_path = ctx->ssd_file.path;
        direct_io_close(&ctx->ssd_file);
        if (ctx->current_session_id == "default") {
            remove(prev_file_path.c_str());
        }
    }

    // 2. Restore the target session if its files exist and validate.
    const std::string new_sess_path = session_file_of(sdir, new_sid);
    const std::string new_kv_path = kv_file_of(ctx, new_sid);

    FILE* test_fp = fopen(new_sess_path.c_str(), "rb");
    if (test_fp) {
        fclose(test_fp);
        direct_io_open(&ctx->ssd_file, new_kv_path.c_str(), false);
        bool loaded = internal_load_session(ctx, new_sess_path);
        if (loaded) {
            reset_sampler(ctx);
            ctx->current_session_id = new_sid;
            fprintf(stderr, "[OPREL SSD] Swapped session to '%s': %zu tokens, %d blocks, active=%d (instant recall).\n",
                    new_sid.c_str(), ctx->history_tokens.size(), (int)ctx->ssd_blocks.size(), ctx->active_tokens);
            prune_sessions(ctx);
            return true;
        }
        if (ctx->ssd_file.is_open) {
            direct_io_close(&ctx->ssd_file);
        }
        // The session was rejected; its block file no longer has a valid index.
        remove(new_kv_path.c_str());
    }

    // 3. Fresh session.
    reset_ctx_state(ctx);
    reset_sampler(ctx);
    if (ctx->quest) quest_free(ctx->quest);
    ctx->quest = quest_create(128, 512);
    direct_io_open(&ctx->ssd_file, new_kv_path.c_str(), true);
    ctx->current_session_id = new_sid;
    fprintf(stderr, "[OPREL SSD] Initialized fresh session '%s'.\n", new_sid.c_str());
    return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[OPREL C++] Exception in oprel_set_session: %s\n", e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[OPREL C++] Unknown exception in oprel_set_session\n");
        return false;
    }
}

OPREL_API void oprel_set_autonomous_restore(OprelContext* ctx, bool enabled) {
    if (ctx) ctx->enable_autonomous_restore = enabled;
}

OPREL_API void oprel_set_retrieval_top_k(OprelContext* ctx, int top_k) {
    if (ctx && top_k > 0) {
        ctx->retrieval_top_k = std::min(8, top_k);
    }
}

OPREL_API int oprel_get_active_restored_blocks(OprelContext* ctx, int32_t* out_block_indices, int max_blocks) {
    if (!ctx || !out_block_indices || max_blocks <= 0) return 0;
    int count = std::min((int)ctx->active_restored_blocks.size(), max_blocks);
    for (int i = 0; i < count; i++) {
        out_block_indices[i] = ctx->active_restored_blocks[i];
    }
    return count;
}


OPREL_API void oprel_free_context(OprelContext* ctx) {
    if (!ctx) return;
    // The caller must not use ctx concurrently with or after this call (the mutex dies with it).
    join_bg_write(ctx);
    if (ctx->total_tokens > 0 && !ctx->current_session_id.empty() && ctx->current_session_id != "default") {
        oprel_save_session(ctx);
    }
    if (ctx->io_aligned_buffer_bg) {
        direct_io_free(ctx->io_aligned_buffer_bg);
        ctx->io_aligned_buffer_bg = NULL;
    }
    if (ctx->ssd_file.is_open) {
        direct_io_close(&ctx->ssd_file);
        // Remove temporary SSD file on context close if default session
        if (ctx->current_session_id == "default") {
            remove(ctx->ssd_file.path);
        }
    }
    if (ctx->io_aligned_buffer) {
        direct_io_free(ctx->io_aligned_buffer);
    }
    if (ctx->quest) {
        quest_free(ctx->quest);
    }
    if (ctx->sampler_chain && g_llama.sampler_free) {
        g_llama.sampler_free(ctx->sampler_chain);
        ctx->sampler_chain = NULL;
    }
    if (ctx->ctx && g_llama.free_context) {
        g_llama.free_context(ctx->ctx);
    }
    delete ctx;
}

OPREL_API void oprel_free_model(OprelModel* model) {
    if (!model) return;
    if (model->model && g_llama.model_free) {
        g_llama.model_free(model->model);
    }
    delete model;
}
