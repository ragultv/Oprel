#include "quest.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

QuestIndex* quest_create(int32_t dim, int32_t initial_capacity) {
    QuestIndex* idx = (QuestIndex*)malloc(sizeof(QuestIndex));
    if (!idx) return NULL;
    idx->dim = dim;
    idx->n_blocks = 0;
    idx->capacity = initial_capacity > 0 ? initial_capacity : 128;
    idx->blocks = (QuestBlockSummary*)malloc(idx->capacity * sizeof(QuestBlockSummary));
    return idx;
}

void quest_free(QuestIndex* idx) {
    if (!idx) return;
    for (int i = 0; i < idx->n_blocks; i++) {
        free(idx->blocks[i].min_k);
        free(idx->blocks[i].max_k);
        free(idx->blocks[i].mean_k);
    }
    free(idx->blocks);
    free(idx);
}

void quest_add_block(
    QuestIndex* idx,
    int32_t block_idx,
    int32_t start_token,
    int32_t n_tokens,
    const float* min_k,
    const float* max_k,
    const float* mean_k
) {
    if (!idx) return;
    if (idx->n_blocks >= idx->capacity) {
        idx->capacity *= 2;
        idx->blocks = (QuestBlockSummary*)realloc(idx->blocks, idx->capacity * sizeof(QuestBlockSummary));
    }
    QuestBlockSummary* b = &idx->blocks[idx->n_blocks++];
    b->block_index = block_idx;
    b->start_token = start_token;
    b->n_tokens = n_tokens;

    b->min_k = (float*)malloc(idx->dim * sizeof(float));
    b->max_k = (float*)malloc(idx->dim * sizeof(float));
    b->mean_k = (float*)malloc(idx->dim * sizeof(float));

    if (min_k) memcpy(b->min_k, min_k, idx->dim * sizeof(float));
    else memset(b->min_k, 0, idx->dim * sizeof(float));

    if (max_k) memcpy(b->max_k, max_k, idx->dim * sizeof(float));
    else memset(b->max_k, 0, idx->dim * sizeof(float));

    if (mean_k) memcpy(b->mean_k, mean_k, idx->dim * sizeof(float));
    else memset(b->mean_k, 0, idx->dim * sizeof(float));
}

struct ScoredBlock {
    int32_t block_index;
    float score;
};

int32_t quest_select_topk(
    const QuestIndex* idx,
    const float* query_vector,
    int32_t top_k,
    int32_t* out_block_indices
) {
    if (!idx || idx->n_blocks == 0 || !out_block_indices || top_k <= 0) {
        return 0;
    }

    std::vector<ScoredBlock> scored;
    scored.reserve(idx->n_blocks);

    for (int i = 0; i < idx->n_blocks; i++) {
        const QuestBlockSummary* b = &idx->blocks[i];
        float score = 0.0f;
        if (query_vector) {
            for (int d = 0; d < idx->dim; d++) {
                float q = query_vector[d];
                // Quest bounding score: max of dot with min or max coordinate
                float s1 = q * b->min_k[d];
                float s2 = q * b->max_k[d];
                score += (s1 > s2 ? s1 : s2);
            }
        } else {
            score = (float)i;
        }
        scored.push_back({b->block_index, score});
    }

    // Sort descending by score
    std::sort(scored.begin(), scored.end(), [](const ScoredBlock& a, const ScoredBlock& b) {
        return a.score > b.score;
    });

    int32_t count = std::min(top_k, (int32_t)scored.size());
    for (int i = 0; i < count; i++) {
        out_block_indices[i] = scored[i].block_index;
    }
    return count;
}

int32_t quest_get_all_scores(
    const QuestIndex* idx,
    const float* query_vector,
    int32_t* out_block_indices,
    float* out_scores,
    int32_t* out_start_tokens,
    int32_t* out_n_tokens,
    int32_t max_blocks
) {
    if (!idx || idx->n_blocks == 0 || max_blocks <= 0) return 0;

    int32_t count = std::min(max_blocks, idx->n_blocks);
    for (int i = 0; i < count; i++) {
        const QuestBlockSummary* b = &idx->blocks[i];
        float score = 0.0f;
        if (query_vector) {
            for (int d = 0; d < idx->dim; d++) {
                float q = query_vector[d];
                float s1 = q * b->min_k[d];
                float s2 = q * b->max_k[d];
                score += (s1 > s2 ? s1 : s2);
            }
        }
        if (out_block_indices) out_block_indices[i] = b->block_index;
        if (out_scores) out_scores[i] = score;
        if (out_start_tokens) out_start_tokens[i] = b->start_token;
        if (out_n_tokens) out_n_tokens[i] = b->n_tokens;
    }
    return count;
}
