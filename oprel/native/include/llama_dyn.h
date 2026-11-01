#ifndef LLAMA_DYN_H
#define LLAMA_DYN_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;

typedef struct llama_model llama_model;
typedef struct llama_context llama_context;
typedef struct llama_vocab llama_vocab;
typedef struct llama_sampler llama_sampler;

typedef struct llama_batch {
    int32_t n_tokens;
    llama_token* token;
    float* embd;
    llama_pos* pos;
    int32_t* n_seq_id;
    llama_seq_id** seq_id;
    int8_t* logits;
} llama_batch;

typedef struct llama_model_params {
    void* devices;
    void* tensor_buft_overrides;
    int32_t n_gpu_layers;
    int32_t split_mode;
    int32_t main_gpu;
    const float* tensor_split;
    void* progress_callback;
    void* progress_callback_user_data;
    const void* kv_overrides;
    bool vocab_only;
    bool use_mmap;
    bool use_direct_io;
    bool use_mlock;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
} llama_model_params;

// Must match `struct llama_context_params` of the pinned llama.cpp build (b9616) field for field;
// llama_ssd_wrapper.cpp static_asserts the offsets it relies on.
typedef struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    int32_t  n_threads;
    int32_t  n_threads_batch;

    int32_t  ctx_type;
    int32_t  rope_scaling_type;
    int32_t  pooling_type;
    int32_t  attention_type;
    int32_t  flash_attn_type;

    float    rope_freq_base;
    float    rope_freq_scale;
    float    yarn_ext_factor;
    float    yarn_attn_factor;
    float    yarn_beta_fast;
    float    yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float    defrag_thold;

    void*    cb_eval;
    void*    cb_eval_user_data;

    int32_t  type_k;
    int32_t  type_v;

    void*    abort_callback;
    void*    abort_callback_data;

    bool     embeddings;
    bool     offload_kqv;
    bool     no_perf;
    bool     op_offload;
    bool     swa_full;
    bool     kv_unified;

    void*    samplers;
    size_t   n_samplers;

    void*    ctx_other;
} llama_context_params;

typedef struct llama_sampler_chain_params {
    bool no_perf;
} llama_sampler_chain_params;

// Function pointers for dynamic loading from llama.dll
typedef void (*fn_llama_backend_init)(void);
typedef void (*fn_llama_backend_free)(void);
typedef llama_model_params (*fn_llama_model_default_params)(void);
typedef llama_model* (*fn_llama_model_load_from_file)(const char* path_model, llama_model_params params);
typedef void (*fn_llama_model_free)(llama_model* model);
typedef const llama_vocab* (*fn_llama_model_get_vocab)(const llama_model* model);
typedef int32_t (*fn_llama_model_n_layer)(const llama_model* model);

typedef llama_context_params (*fn_llama_context_default_params)(void);
typedef llama_context* (*fn_llama_new_context_with_model)(llama_model* model, llama_context_params params);
typedef void (*fn_llama_free)(llama_context* ctx);

typedef int32_t (*fn_llama_tokenize)(
    const llama_vocab* vocab,
    const char* text,
    int32_t text_len,
    llama_token* tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
);

typedef int32_t (*fn_llama_token_to_piece)(
    const llama_vocab* vocab,
    llama_token token,
    char* buf,
    int32_t length,
    int32_t lstrip,
    bool special
);

typedef bool (*fn_llama_token_is_eog)(const llama_vocab* vocab, llama_token token);

typedef llama_batch (*fn_llama_batch_init)(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
typedef llama_batch (*fn_llama_batch_get_one)(llama_token* tokens, int32_t n_tokens);
typedef void (*fn_llama_batch_free)(llama_batch batch);
typedef int32_t (*fn_llama_decode)(llama_context* ctx, llama_batch batch);
typedef float* (*fn_llama_get_logits_ith)(llama_context* ctx, int32_t i);

typedef llama_sampler_chain_params (*fn_llama_sampler_chain_default_params)(void);
typedef llama_sampler* (*fn_llama_sampler_chain_init)(llama_sampler_chain_params params);
typedef void (*fn_llama_sampler_chain_add)(llama_sampler* chain, llama_sampler* smpl);
typedef llama_sampler* (*fn_llama_sampler_init_greedy)(void);
typedef llama_sampler* (*fn_llama_sampler_init_dist)(uint32_t seed);
typedef llama_sampler* (*fn_llama_sampler_init_temp)(float t);
typedef llama_sampler* (*fn_llama_sampler_init_top_p)(float p, size_t min_keep);
typedef llama_sampler* (*fn_llama_sampler_init_top_k)(int32_t k);
typedef llama_token (*fn_llama_vocab_eos)(const llama_vocab* vocab);
typedef llama_token (*fn_llama_vocab_eot)(const llama_vocab* vocab);
typedef llama_token (*fn_llama_sampler_sample)(llama_sampler* smpl, llama_context* ctx, int32_t idx);
typedef void (*fn_llama_sampler_free)(llama_sampler* smpl);

typedef void* llama_memory_t;
typedef llama_memory_t (*fn_llama_get_memory)(const llama_context* ctx);
typedef void (*fn_llama_memory_clear)(llama_memory_t mem, bool data);
typedef bool (*fn_llama_memory_seq_rm)(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
typedef void (*fn_llama_memory_seq_cp)(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
typedef void (*fn_llama_memory_seq_add)(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
typedef llama_pos (*fn_llama_memory_seq_pos_max)(llama_memory_t mem, llama_seq_id seq_id);
typedef llama_pos (*fn_llama_memory_seq_pos_min)(llama_memory_t mem, llama_seq_id seq_id);
typedef size_t (*fn_llama_state_seq_get_size_ext)(llama_context* ctx, llama_seq_id seq_id, uint32_t flags);
typedef size_t (*fn_llama_state_seq_get_data_ext)(llama_context* ctx, uint8_t* dst, size_t size, llama_seq_id seq_id, uint32_t flags);
typedef size_t (*fn_llama_state_seq_set_data_ext)(llama_context* ctx, const uint8_t* src, size_t size, llama_seq_id seq_id, uint32_t flags);
typedef int32_t (*fn_llama_vocab_n_tokens)(const llama_vocab* vocab);
typedef void (*fn_llama_sampler_accept)(llama_sampler* smpl, llama_token token);
typedef llama_sampler* (*fn_llama_sampler_init_penalties)(
    int32_t penalty_last_n,
    float   penalty_repeat,
    float   penalty_freq,
    float   penalty_present
);
typedef llama_sampler* (*fn_llama_sampler_init_dry)(
    const llama_vocab* vocab,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_penalty_last_n,
    const char** seq_breakers,
    size_t num_breakers
);

typedef struct LlamaFunctions {
    fn_llama_backend_init backend_init;
    fn_llama_backend_free backend_free;
    fn_llama_model_default_params model_default_params;
    fn_llama_model_load_from_file model_load_from_file;
    fn_llama_model_free model_free;
    fn_llama_model_get_vocab model_get_vocab;
    fn_llama_model_n_layer model_n_layer;

    fn_llama_context_default_params context_default_params;
    fn_llama_new_context_with_model new_context_with_model;
    fn_llama_free free_context;

    fn_llama_tokenize tokenize;
    fn_llama_token_to_piece token_to_piece;
    fn_llama_token_is_eog token_is_eog;
    fn_llama_vocab_n_tokens vocab_n_tokens;

    fn_llama_batch_init batch_init;
    fn_llama_batch_get_one batch_get_one;
    fn_llama_batch_free batch_free;
    fn_llama_decode decode;
    fn_llama_get_logits_ith get_logits_ith;

    fn_llama_sampler_chain_default_params sampler_chain_default_params;
    fn_llama_sampler_chain_init sampler_chain_init;
    fn_llama_sampler_chain_add sampler_chain_add;
    fn_llama_sampler_init_greedy sampler_init_greedy;
    fn_llama_sampler_init_dist sampler_init_dist;
    fn_llama_sampler_init_temp sampler_init_temp;
    fn_llama_sampler_init_top_p sampler_init_top_p;
    fn_llama_sampler_init_top_k sampler_init_top_k;
    fn_llama_vocab_eos vocab_eos;
    fn_llama_vocab_eot vocab_eot;
    fn_llama_sampler_init_penalties sampler_init_penalties;
    fn_llama_sampler_init_dry sampler_init_dry;
    fn_llama_sampler_accept sampler_accept;
    fn_llama_sampler_sample sampler_sample;
    fn_llama_sampler_free sampler_free;

    fn_llama_get_memory get_memory;
    fn_llama_memory_clear memory_clear;
    fn_llama_memory_seq_rm memory_seq_rm;
    fn_llama_memory_seq_cp memory_seq_cp;
    fn_llama_memory_seq_add memory_seq_add;
    fn_llama_memory_seq_pos_max memory_seq_pos_max;
    fn_llama_memory_seq_pos_min memory_seq_pos_min;
    fn_llama_state_seq_get_size_ext state_seq_get_size_ext;
    fn_llama_state_seq_get_data_ext state_seq_get_data_ext;
    fn_llama_state_seq_set_data_ext state_seq_set_data_ext;
} LlamaFunctions;

extern LlamaFunctions g_llama;
bool load_llama_library(const char* dll_path);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_DYN_H
