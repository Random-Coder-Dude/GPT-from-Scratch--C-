#include <string.h>
#include "imports.h"

char* run_transformer_model(
    const char** vocab,
    const char* input_text,
    int embedding_dim, int hidden_dim, int num_heads
) {
    int vocab_size = sizeof(vocab);
    srand((unsigned int)time(NULL));

    // Tokenize
    int token_ids[32];
    int seq_len = tokenize(input_text, token_ids);
    if (seq_len > 32) {
        fprintf(stderr, "Error: Sequence too long.\n");
        return NULL;
    }

    // Embedding
    Embedding* emb = createEmbedding(vocab_size, embedding_dim);
    if (!emb) return NULL;

    Matrix* embedded = embedTokens(emb, token_ids, seq_len);
    if (!embedded) {
        freeEmbedding(emb);
        return NULL;
    }

    // Positional Encoding
    Matrix* pos = createPositionalEncoding(seq_len, embedding_dim);
    if (!pos) {
        freeEmbedding(emb);
        freeMatrix(embedded);
        return NULL;
    }

    for (int i = 0; i < embedded->rows * embedded->columns; i++) {
        embedded->data[i] += pos->data[i];
    }

    // Transformer Block
    TransformerBlock* block = createTransformerBlock(embedding_dim, num_heads, hidden_dim);
    if (!block) {
        freeEmbedding(emb);
        freeMatrix(embedded);
        freeMatrix(pos);
        return NULL;
    }

    Matrix* transformer_output = transformerForward(block, embedded);
    if (!transformer_output) {
        freeEmbedding(emb);
        freeMatrix(embedded);
        freeMatrix(pos);
        freeTransformerBlock(block);
        return NULL;
    }

    // Output Projection
    OutputProjection* out_proj = createOutputProjection(embedding_dim, vocab_size);
    if (!out_proj) {
        freeEmbedding(emb);
        freeMatrix(embedded);
        freeMatrix(pos);
        freeMatrix(transformer_output);
        freeTransformerBlock(block);
        return NULL;
    }

    Matrix* logits = outputForward(out_proj, transformer_output);
    if (!logits) {
        freeEmbedding(emb);
        freeMatrix(embedded);
        freeMatrix(pos);
        freeMatrix(transformer_output);
        freeTransformerBlock(block);
        freeOutputProjection(out_proj);
        return NULL;
    }

    // Convert token IDs to words
    size_t output_buffer_size = seq_len * 32;
    char* output_string = (char*)malloc(output_buffer_size);
    if (!output_string) {
        return NULL;
    }
    output_string[0] = '\0';

    for (int i = 0; i < logits->rows; i++) {
        int token_id = argmaxRow(logits, i);
        strncat(output_string, vocab[token_id], output_buffer_size - strlen(output_string) - 1);
        strncat(output_string, " ", output_buffer_size - strlen(output_string) - 1);
    }

    // Cleanup
    freeEmbedding(emb);
    freeMatrix(embedded);
    freeMatrix(pos);
    freeTransformerBlock(block);
    freeOutputProjection(out_proj);
    freeMatrix(logits);

    return output_string;
}
