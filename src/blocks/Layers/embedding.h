#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "matrixUtils.h"

typedef struct {
    int vocab_size;
    int embedding_dim;
    Matrix* table;
} Embedding;

Embedding* createEmbedding(int vocab_size, int embedding_dim);
Matrix* embedTokens(Embedding* embedding, int* token_ids, int length);
void freeEmbedding(Embedding* embedding);

#endif
