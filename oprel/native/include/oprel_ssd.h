#ifndef OPREL_SSD_H
#define OPREL_SSD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
  #ifdef OPREL_BUILD_DLL
    #define OPREL_API __declspec(dllexport)
  #else
    #define OPREL_API __declspec(dllimport)
  #endif
#else
  #define OPREL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OprelModel OprelModel;
typedef struct OprelContext OprelContext;

#pragma pack(push, 1)
typedef struct OprelSSDBlockHeader {
    char     magic[8];       // "OPREL_KV"
    uint32_t format_version; // 2
    uint32_t llama_build;    // 9616
    uint8_t  model_sha256[32]; // SHA-256 of the model artifact
    uint8_t  runtime_sha256[32]; // SHA-256 of the loaded llama runtime DLL
    int32_t  n_layer;        // layer count
    int32_t  type_k;         // GGML_TYPE_Q4_0 = 2
    int32_t  type_v;         // GGML_TYPE_Q4_0 = 2
    int32_t  p0;             // range start
    int32_t  p1;             // range end
    int32_t  n_tokens;       // tokens count
    uint64_t state_bytes;    // payload byte size
    uint32_t checksum;       // CRC32 checksum
    uint32_t reserved;
} OprelSSDBlockHeader;
#pragma pack(pop)

static_assert(sizeof(OprelSSDBlockHeader) == 120, "OPREL_KV header layout changed");

typedef struct OprelStats {
    int32_t total_tokens;
    int32_t active_tokens;
    int32_t ssd_blocks;
    int32_t retrieved_blocks;
    double  last_retrieval_ms;
    double  last_read_ms;
    double  last_set_data_ms;
    double  read_throughput_mb_s;
    int64_t decode_calls;
    int64_t last_block_bytes;
    int64_t total_ssd_bytes_written;
    int64_t total_ssd_bytes_read;
    int64_t archive_validation_passes;
    int64_t archive_validation_failures;
    int64_t restore_decode_calls;
    int64_t restore_prefill_tokens;
    int32_t last_restore_pos_start;
    int32_t last_restore_pos_end;
    int32_t last_restore_kv_match;
    double  last_restore_latency_ms;
} OprelStats;

/**
 * Initialize dynamic binding to the target llama.dll.
 * Returns true if all required symbols are successfully resolved.
 */
OPREL_API bool oprel_init(const char* llama_dll_path);

/**
 * Load a GGUF model through the bound llama.dll.
 */
OPREL_API OprelModel* oprel_load_model(const char* model_path, int n_gpu_layers);

/**
 * Create an SSD-backed context.
 * active_ctx_size: Size of active VRAM window (e.g. 2048 or 4096 tokens).
 * max_ssd_context: Maximum context limit on SSD (e.g. 131072 tokens).
 */
OPREL_API OprelContext* oprel_create_context(
    OprelModel* model,
    const char* ssd_cache_dir,
    int active_ctx_size,
    int max_ssd_context
);

/**
 * Tokenize and ingest a prompt string into the context and SSD store.
 */
OPREL_API int oprel_ingest_prompt(
    OprelContext* ctx,
    const char* prompt,
    bool reset_context
);

/**
 * Evaluate next token generation step with Quest SSD retrieval.
 * Returns the generated token ID, or -1 on EOS / error.
 */
OPREL_API int oprel_generate_next_token(
    OprelContext* ctx,
    float temperature,
    float top_p
);

OPREL_API int oprel_generate_next_token_ex(
    OprelContext* ctx,
    float temperature,
    float top_p,
    float repeat_penalty,
    float freq_penalty,
    float presence_penalty
);

/**
 * Detokenize a token ID into text.
 */
OPREL_API int oprel_token_to_text(
    OprelContext* ctx,
    int token,
    char* out_buf,
    int max_len
);

/**
 * Check if token is End of Generation / Sequence.
 */
OPREL_API bool oprel_is_eog(OprelContext* ctx, int token);

/**
 * Retrieve relevant historical context from SSD using Quest index.
 * Reads the sector-aligned KV state from SSD via Direct I/O and returns retrieved text.
 */
OPREL_API int oprel_retrieve_context(
    OprelContext* ctx,
    const char* query,
    int top_k,
    char* out_buf,
    int max_len
);

/**
 * Restore an archived KV block from SSD into the live window using llama_state_seq_set_data_ext.
 * The block's tokens are never decoded: only the serialised KV bytes are injected. The block is
 * placed after the attention sinks (and earlier restored blocks), the live tail shifts right, and
 * one single-token decode of the last live token applies the lazy K-shift and refreshes the
 * logits so the next sample attends to the restored prefix.
 */
OPREL_API bool oprel_restore_kv_state(
    OprelContext* ctx,
    int block_index
);

/**
 * Like oprel_restore_kv_state, with control over the logits refresh. With refresh_logits=false the
 * block is injected but the K-shift stays pending until the next decode, and the logits still
 * describe the cache *without* the block: only use it when a decode follows before sampling.
 */
OPREL_API bool oprel_restore_kv_block(OprelContext* ctx, int block_index, bool refresh_logits);

/**
 * Archive the oldest `n_tokens` live tokens (right after the attention sinks) to SSD now, instead
 * of waiting for the automatic window trigger. Returns the new block index, or -1 on failure.
 */
OPREL_API int oprel_archive_oldest(OprelContext* ctx, int n_tokens);

/**
 * Recompute the logits of the last live token against the current cache by re-decoding that one
 * token. This also applies any pending K-shift. Returns false if the decode failed.
 */
OPREL_API bool oprel_refresh_logits(OprelContext* ctx);

/**
 * Identify the best matching SSD block index for a query.
 */
OPREL_API int oprel_find_best_block(
    OprelContext* ctx,
    const char* query
);

/**
 * Score all archived SSD blocks against a query for ranker evaluation.
 */
OPREL_API int oprel_score_all_blocks(
    OprelContext* ctx,
    const char* query,
    int32_t* out_block_indices,
    float* out_scores,
    int32_t* out_start_tokens,
    int32_t* out_n_tokens,
    int max_blocks
);

/**
 * Retrieve live statistics of SSD offload and retrieval.
 */
OPREL_API void oprel_get_stats(OprelContext* ctx, OprelStats* out_stats);

/**
 * Retrieve logits of the last decoded token across vocabulary.
 * Returns number of floats copied to out_logits.
 */
OPREL_API int oprel_get_logits(OprelContext* ctx, float* out_logits, int max_vocab);

/**
 * Re-serialize token range [p0, p1) from sequence 0 via scratch sequence 1.
 * Returns size in bytes of the re-serialized payload.
 */
OPREL_API int oprel_reserialize_range(OprelContext* ctx, int p0, int p1, uint8_t* out_buf, int max_len);

/**
 * Read the raw stored payload of block_index directly from SSD.
 * Returns size in bytes of the stored payload.
 */
OPREL_API int oprel_get_block_payload(OprelContext* ctx, int block_index, uint8_t* out_buf, int max_len);

/**
 * Switch active conversation session.
 * Automatically persists the previous session's live KV and SSD registry,
 * and loads or initializes the target session without cold re-prefill.
 */
OPREL_API bool oprel_set_session(OprelContext* ctx, const char* session_id);

/**
 * Persist the current active session state (live KV and SSD block index) to disk.
 */
OPREL_API bool oprel_save_session(OprelContext* ctx);

/**
 * Enable or disable autonomous query-time SSD block restoration.
 */
OPREL_API void oprel_set_autonomous_restore(OprelContext* ctx, bool enabled);

/**
 * Set the number of top-K historical blocks autonomously restored from SSD.
 * Default is 2 blocks (1024 tokens).
 */
OPREL_API void oprel_set_retrieval_top_k(OprelContext* ctx, int top_k);

/**
 * Get active restored block indices.
 * Returns number of active restored blocks copied to out_block_indices.
 */
OPREL_API int oprel_get_active_restored_blocks(OprelContext* ctx, int32_t* out_block_indices, int max_blocks);

/**
 * Cleanup and free context and model.
 */
OPREL_API void oprel_free_context(OprelContext* ctx);
OPREL_API void oprel_free_model(OprelModel* model);

#ifdef __cplusplus
}
#endif

#endif // OPREL_SSD_H
