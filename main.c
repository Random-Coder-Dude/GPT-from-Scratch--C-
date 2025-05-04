#include "src/imports.h"

const char* vocab[100] = {
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
    "hello", "world", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "[PAD]", "[UNK]", "[SEP]", "[CLS]", ".", ",", "!", "?", "-", "+", "*", "(", ")"
    // Add more up to 100 if needed
};

int main() {
    srand((unsigned int)time(NULL));

    // Start timing the whole program
    clock_t start_time = clock();

    // Tokenize input text
    const char* text = "the quick brown fox ate my cat";
    printf("%s\n", text);
    int token_ids[32];
    int seq_len = tokenize(text, token_ids);

    if (seq_len > 32) {
        printf("Error: Sequence length exceeds array size.\n");
        return 1;
    }

    // Record time after tokenization
    clock_t tokenize_time = clock();

    // Model hyperparameters
    int vocab_size = 50;
    int embedding_dim = 768;
    int hidden_dim = 768;
    int num_heads = 96;

    // Embedding
    Embedding* emb = createEmbedding(vocab_size, embedding_dim);
    if (!emb) {
        printf("Error: Embedding creation failed.\n");
        return 1;
    }
    Matrix* embedded = embedTokens(emb, token_ids, seq_len);  // (seq_len × embedding_dim)
    if (!embedded) {
        printf("Error: Embedding tokens failed.\n");
        return 1;
    }

    // Record time after embedding
    clock_t embed_time = clock();

    // Positional Encoding
    Matrix* pos = createPositionalEncoding(seq_len, embedding_dim);
    if (!pos) {
        printf("Error: Positional encoding creation failed.\n");
        return 1;
    }

    // Record time after positional encoding
    clock_t pos_time = clock();

    // Add position to embeddings
    for (int i = 0; i < embedded->rows * embedded->columns; i++) {
        embedded->data[i] += pos->data[i];
    }

    // Transformer Block
    TransformerBlock* block = createTransformerBlock(embedding_dim, num_heads, hidden_dim);
    if (!block) {
        printf("Error: Transformer block creation failed.\n");
        return 1;
    }
    Matrix* transformer_output = transformerForward(block, embedded);  // (seq_len × embedding_dim)
    if (!transformer_output) {
        printf("Error: Transformer forward pass failed.\n");
        return 1;
    }

    // Record time after transformer block
    clock_t transformer_time = clock();

    // Output Projection
    OutputProjection* out_proj = createOutputProjection(embedding_dim, vocab_size);
    if (!out_proj) {
        printf("Error: Output projection creation failed.\n");
        return 1;
    }

    Matrix* logits = outputForward(out_proj, transformer_output);  // (seq_len × vocab_size)
    if (!logits) {
        printf("Error: Output forward pass failed.\n");
        return 1;
    }

    // Record time after output projection
    clock_t output_time = clock();

    for (int i = 0; i < logits->rows; i++) {
        int token_id = argmaxRow(logits, i);
        printf("%s ", vocab[token_id]);
    }

    // Record time after printing the result
    clock_t end_time = clock();

    // Calculate and print the time taken for each step
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double tokenize_duration = (double)(tokenize_time - start_time) / CLOCKS_PER_SEC;
    double embed_duration = (double)(embed_time - tokenize_time) / CLOCKS_PER_SEC;
    double pos_duration = (double)(pos_time - embed_time) / CLOCKS_PER_SEC;
    double transformer_duration = (double)(transformer_time - pos_time) / CLOCKS_PER_SEC;
    double output_duration = (double)(output_time - transformer_time) / CLOCKS_PER_SEC;
    double result_duration = (double)(end_time - output_time) / CLOCKS_PER_SEC;

    printf("\nTime taken for each step:\n");
    printf("Total execution time: %.6f seconds\n", total_time);
    printf("Tokenization time: %.6f seconds\n", tokenize_duration);
    printf("Embedding time: %.6f seconds\n", embed_duration);
    printf("Positional Encoding time: %.6f seconds\n", pos_duration);
    printf("Transformer Block time: %.6f seconds\n", transformer_duration);
    printf("Output Projection time: %.6f seconds\n", output_duration);
    printf("Result Printing time: %.6f seconds\n", result_duration);

    // Cleanup
    freeMatrix(embedded);
    freeMatrix(pos);
    freeMatrix(transformer_output);
    freeEmbedding(emb);
    freeTransformerBlock(block);
    freeOutputProjection(out_proj);

    return 0;  // Ensure success return
}
