#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "matrixUtils.h"
#include "multiHeadAttention.h"
#include "ffn.h"

typedef struct {
    MultiHeadAttention* mha;
    FeedForward* ffn;
} TransformerBlock;

TransformerBlock* createTransformerBlock(int input_dim, int num_heads, int hidden_dim);
Matrix* transformerForward(TransformerBlock* block, Matrix* input);
void freeTransformerBlock(TransformerBlock* block);

#endif
