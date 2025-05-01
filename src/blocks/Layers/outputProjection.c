#include <stdio.h>
#include <stdlib.h>
#include "outputProjection.h"

OutputProjection* createOutputProjection(int input_dim, int vocab_size) {
    OutputProjection* out = (OutputProjection*)malloc(sizeof(OutputProjection));
    out->proj = createLinearLayer(input_dim, vocab_size);
    return out;
}

// input: (seq_len × input_dim)
// returns: (seq_len × vocab_size)
Matrix* outputForward(OutputProjection* out_proj, Matrix* input) {
    // Transpose: (seq_len × dim) → (dim × seq_len)
    Matrix* input_T = transposeMatrix(input);

    // Multiply: W (vocab_size × dim) × input_T (dim × seq_len) → (vocab_size × seq_len)
    Matrix* logits_T = multiplyMatrix(out_proj->proj->weights, input_T);
    freeMatrix(input_T);

    // Add bias
    for (int i = 0; i < logits_T->rows; i++) {
        for (int j = 0; j < logits_T->columns; j++) {
            logits_T->data[i * logits_T->columns + j] += out_proj->proj->bias->data[i];
        }
    }

    // Transpose back: (vocab_size × seq_len) → (seq_len × vocab_size)
    Matrix* logits = transposeMatrix(logits_T);
    freeMatrix(logits_T);

    return logits;
}

void freeOutputProjection(OutputProjection* out_proj) {
    freeLinearLayer(out_proj->proj);
    free(out_proj);
}
