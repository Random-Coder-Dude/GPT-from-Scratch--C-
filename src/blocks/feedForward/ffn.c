#include <stdlib.h>
#include "ffn.h"
#include "matrixUtils.h"

static void reluInPlace(Matrix* matrix) {
    for (int i = 0; i < matrix->rows * matrix->columns; i++) {
        if (matrix->data[i] < 0.0f) {
            matrix->data[i] = 0.0f;
        }
    }
}

FeedForward* createFeedForward(int input_dim, int hidden_dim) {
    FeedForward* ffn = (FeedForward*)malloc(sizeof(FeedForward));
    ffn->fc1 = createLinearLayer(input_dim, hidden_dim);
    ffn->fc2 = createLinearLayer(hidden_dim, input_dim);
    return ffn;
}

Matrix* feedForwardForward(FeedForward* ffn, Matrix* input) {
    // Layer 1
    Matrix* h1 = multiplyMatrix(ffn->fc1->weights, input); // (hidden_dim × batch_size)

    // Add bias
    for (int i = 0; i < h1->rows; i++) {
        for (int j = 0; j < h1->columns; j++) {
            h1->data[i * h1->columns + j] += ffn->fc1->bias->data[i];
        }
    }

    Matrix* h1_T = transposeMatrix(h1); // (batch_size × hidden_dim)
    freeMatrix(h1);
    reluInPlace(h1_T); // In-place ReLU

    // Layer 2
    Matrix* h2 = multiplyMatrix(ffn->fc2->weights, transposeMatrix(h1_T)); // (input_dim × batch_size)
    freeMatrix(h1_T);

    // Add bias
    for (int i = 0; i < h2->rows; i++) {
        for (int j = 0; j < h2->columns; j++) {
            h2->data[i * h2->columns + j] += ffn->fc2->bias->data[i];
        }
    }

    return h2; // output shape: (input_dim × batch_size)
}

void freeFeedForward(FeedForward* ffn) {
    freeLinearLayer(ffn->fc1);
    freeLinearLayer(ffn->fc2);
    free(ffn);
}
