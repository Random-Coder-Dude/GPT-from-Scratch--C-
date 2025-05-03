#include <stdio.h>
#include <stdlib.h>
#include "outputProjection.h"

OutputProjection* createOutputProjection(int input_dim, int vocab_size) {
    OutputProjection* out = (OutputProjection*)malloc(sizeof(OutputProjection));
    out->proj = createLinearLayer(input_dim, vocab_size);
    return out;
}

Matrix* outputForward(OutputProjection* out_proj, Matrix* input) {
    Matrix* logits_T = multiplyMatrix(out_proj->proj->weights, input);
    freeMatrix(input);

    for (int col = 0; col < logits_T->columns; col++) {
        for (int row = 0; row < logits_T->rows; row++) {
            logits_T->data[col * logits_T->rows + row] += out_proj->proj->bias->data[row];
        }
    }

    Matrix* logits = transposeMatrix(logits_T);
    freeMatrix(logits_T);
    return logits;
}


void freeOutputProjection(OutputProjection* out_proj) {
    freeLinearLayer(out_proj->proj);
    free(out_proj);
}
