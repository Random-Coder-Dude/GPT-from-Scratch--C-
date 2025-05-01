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

    // Tokenize input text
    const char* text = "the quick brown fox";
    printf("%s\n", text);
    int token_ids[32];
    int seq_len = tokenize(text, token_ids);

    // Model hyperparameters
    int vocab_size = 50;
    int embedding_dim = 4;
    int hidden_dim = 16;
    int num_heads = 2;

    // Embedding
    Embedding* emb = createEmbedding(vocab_size, embedding_dim);
    Matrix* embedded = embedTokens(emb, token_ids, seq_len);  // (seq_len × embedding_dim)

    // Positional Encoding
    Matrix* pos = createPositionalEncoding(seq_len, embedding_dim);

    // Add position to embeddings
    for (int i = 0; i < embedded->rows * embedded->columns; i++) {
        embedded->data[i] += pos->data[i];
    }

    // Transformer Block
    TransformerBlock* block = createTransformerBlock(embedding_dim, num_heads, hidden_dim);
    Matrix* transformer_output = transformerForward(block, embedded);  // (seq_len × embedding_dim)

    // Output Projection
    OutputProjection* out_proj = createOutputProjection(embedding_dim, vocab_size);

    Matrix* logits = outputForward(out_proj, transformer_output);  // (seq_len × vocab_size)

    for (int i = 0; i < logits->rows; i++) {
        int token_id = argmaxRow(logits, i);
        printf("%s ", vocab[token_id]);
    }

    // Cleanup
    freeMatrix(embedded);
    freeMatrix(pos);
    freeMatrix(transformer_output);
    freeMatrix(logits);
    freeEmbedding(emb);
    freeTransformerBlock(block);
    freeOutputProjection(out_proj);

    return 0;
}
