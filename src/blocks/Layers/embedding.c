#include <stdlib.h>
#include <stdio.h>
#include "embedding.h"

Embedding* createEmbedding(int vocab_size, int embedding_dim) {
    Embedding* emb = (Embedding*)malloc(sizeof(Embedding));
    emb->vocab_size = vocab_size;
    emb->embedding_dim = embedding_dim;
    emb->table = createMatrix(vocab_size, embedding_dim);

    // Random init: e.g. [-0.5, 0.5)
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        emb->table->data[i] = ((float)rand() / RAND_MAX - 0.5f);
    }

    return emb;
}

// Returns a (embedding_dim × length) matrix in column-major order
Matrix* embedTokens(Embedding* embedding, int* token_ids, int length) {
    Matrix* result = createMatrix(embedding->embedding_dim, length);  // rows = embedding_dim, cols = length

    for (int i = 0; i < length; i++) {
        int token_id = token_ids[i];

        if (token_id < 0 || token_id >= embedding->vocab_size) {
            printf("Error: Token ID out of bounds\n");
            exit(1);
        }

        for (int j = 0; j < embedding->embedding_dim; j++) {
            // Column-major access: result[j][i]
            result->data[i * embedding->embedding_dim + j] =  // or j * length + i depending on layout
                embedding->table->data[j * embedding->vocab_size + token_id];
        }
    }

    return result;
}

void freeEmbedding(Embedding* embedding) {
    freeMatrix(embedding->table);
    free(embedding);
}
