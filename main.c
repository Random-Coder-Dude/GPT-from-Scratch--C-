#include "src/imports.h"

int main() {
    srand((unsigned int)time(NULL));

    // Input text
    const char* text = "Hello TESTING!!!!";
    int token_ids[32];  // Output buffer

    // Tokenize
    int seq_len = tokenize(text, token_ids);
    printf("Token IDs (%d tokens): ", seq_len);
    for (int i = 0; i < seq_len; i++) {
        printf("%d ", token_ids[i]);
    }
    printf("\n");

    // Embedding
    int vocab_size = 100;     // dummy limit
    int embedding_dim = 8;
    Embedding* emb = createEmbedding(vocab_size, embedding_dim);
    Matrix* embedded = embedTokens(emb, token_ids, seq_len);

    // Positional encoding
    Matrix* pos = createPositionalEncoding(seq_len, embedding_dim);

    // Add embeddings + position encoding
    for (int i = 0; i < embedded->rows * embedded->columns; i++) {
        embedded->data[i] += pos->data[i];
    }

    printf("Embedded + Positional:\n");
    printMatrix(embedded);

    // Cleanup
    freeMatrix(pos);
    freeMatrix(embedded);
    freeEmbedding(emb);
    return 0;
}
