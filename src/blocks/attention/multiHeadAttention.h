#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "matrixUtils.h"
#include "linearLayer.h"

typedef struct {
    int num_heads;
    int input_dim;
    int head_dim;
    LinearLayer** Wq;
    LinearLayer** Wk;
    LinearLayer** Wv;
    LinearLayer* Wo;
} MultiHeadAttention;

MultiHeadAttention* createMultiHeadAttention(int input_dim, int num_heads);
Matrix* multiHeadAttentionForward(MultiHeadAttention* mha, Matrix* input);
void freeMultiHeadAttention(MultiHeadAttention* mha);

#endif
