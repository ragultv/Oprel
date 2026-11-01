#ifndef QUEST_H
#define QUEST_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QuestBlockSummary {
    int32_t block_index;
    int32_t start_token;
    int32_t n_tokens;
    float* min_k;     // length: dim
    float* max_k;     // length: dim
    float* mean_k;    // length: dim
} QuestBlockSummary;

typedef struct QuestIndex {
    int32_t dim;
    int32_t n_blocks;
    int32_t capacity;
    QuestBlockSummary* blocks;
} QuestIndex;

QuestIndex* quest_create(int32_t dim, int32_t initial_capacity);
void quest_free(QuestIndex* idx);

/**
 * Add a new block summary to the Quest index.
 */
void quest_add_block(
    QuestIndex* idx,
    int32_t block_idx,
    int32_t start_token,
    int32_t n_tokens,
    const float* min_k,
    const float* max_k,
    const float* mean_k
);

/**
 * Compute top-K relevant block indices for the current query vector.
 * out_block_indices must have capacity for at least top_k ints.
 * Returns the actual number of blocks selected (min(top_k, n_blocks)).
 */
int32_t quest_select_topk(
    const QuestIndex* idx,
    const float* query_vector,
    int32_t top_k,
    int32_t* out_block_indices
);

/**
 * Score all blocks and return indices, scores, and token boundaries.
 */
int32_t quest_get_all_scores(
    const QuestIndex* idx,
    const float* query_vector,
    int32_t* out_block_indices,
    float* out_scores,
    int32_t* out_start_tokens,
    int32_t* out_n_tokens,
    int32_t max_blocks
);

#ifdef __cplusplus
}
#endif

#endif // QUEST_H
