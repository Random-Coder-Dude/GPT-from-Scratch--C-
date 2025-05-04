#ifndef GPT_H
#define GPT_H

char* run_transformer_model(
    const char** vocab,
    const char* input_text,
    int embedding_dim, int hidden_dim, int num_heads
);

#endif
