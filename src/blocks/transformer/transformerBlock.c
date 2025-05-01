#include <stdio.h>
#include <stdlib.h>
#include "transformerBlock.h"
#include "normalization.h"
#include "matrixUtils.h"

Matrix* residualAdd(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->columns != b->columns) {
        exit(1);
    }

    Matrix* out = createMatrix(a->rows, a->columns);
    for (int i = 0; i < a->rows * a->columns; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

TransformerBlock* createTransformerBlock(int input_dim, int num_heads, int hidden_dim) {
    TransformerBlock* block = (TransformerBlock*)malloc(sizeof(TransformerBlock));
    block->mha = createMultiHeadAttention(input_dim, num_heads);
    block->ffn = createFeedForward(input_dim, hidden_dim);
    return block;
}

Matrix* transformerForward(TransformerBlock* block, Matrix* input) {
    // 1. Multi-Head Attention
    Matrix* attn_output = multiHeadAttentionForward(block->mha, input);

    // 2. Residual + LayerNorm
    Matrix* mha_residual = residualAdd(attn_output, input);
    freeMatrix(attn_output);
    layerNormalization(mha_residual, 1e-5f);

    // 3. Feed Forward Network
    Matrix* ffn_output = feedForwardForward(block->ffn, mha_residual);

    // 4. Residual + LayerNorm again
    Matrix* final = residualAdd(ffn_output, mha_residual);

    freeMatrix(ffn_output);
    freeMatrix(mha_residual);
    layerNormalization(final, 1e-5f);

    return final;
}

void freeTransformerBlock(TransformerBlock* block) {
    freeMultiHeadAttention(block->mha);
    freeFeedForward(block->ffn);
    free(block);
}
