#include <stdio.h>
#include <stdlib.h>
#include "multiHeadAttention.h"
#include "attentionHead.h"

MultiHeadAttention* createMultiHeadAttention(int input_dim, int num_heads) {
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    mha->num_heads = num_heads;
    mha->input_dim = input_dim;
    mha->head_dim = input_dim / num_heads;

    mha->Wq = (LinearLayer**)malloc(num_heads * sizeof(LinearLayer*));
    mha->Wk = (LinearLayer**)malloc(num_heads * sizeof(LinearLayer*));
    mha->Wv = (LinearLayer**)malloc(num_heads * sizeof(LinearLayer*));

    for (int i = 0; i < num_heads; i++) {
        mha->Wq[i] = createLinearLayer(input_dim, mha->head_dim);
        mha->Wk[i] = createLinearLayer(input_dim, mha->head_dim);
        mha->Wv[i] = createLinearLayer(input_dim, mha->head_dim);
    }

    mha->Wo = createLinearLayer(input_dim, input_dim);

    return mha;
}

Matrix* multiHeadAttentionForward(MultiHeadAttention* mha, Matrix* input) {

    int batch_size = input->columns; // Input shape: (input_dim, batch_size)

    Matrix** heads = (Matrix**)malloc(mha->num_heads * sizeof(Matrix*));

    for (int i = 0; i < mha->num_heads; i++) {
        Matrix* input_T = transposeMatrix(input);

        Matrix* Q = multiplyMatrix(mha->Wq[i]->weights, input_T); // (head_dim, batch_size)
        Matrix* K = multiplyMatrix(mha->Wk[i]->weights, input_T);
        Matrix* V = multiplyMatrix(mha->Wv[i]->weights, input_T);

        // Transpose to (batch_size, head_dim)
        Matrix* Q_T = transposeMatrix(Q);
        Matrix* K_T = transposeMatrix(K);
        Matrix* V_T = transposeMatrix(V);

        heads[i] = attentionHead(Q_T, K_T, V_T); // Output: (batch_size, head_dim)

        // Free intermediates
        freeMatrix(Q);
        freeMatrix(K);
        freeMatrix(V);
        freeMatrix(Q_T);
        freeMatrix(K_T);
        freeMatrix(V_T);
    }

    // Concatenate heads horizontally
    int concat_cols = heads[0]->columns * mha->num_heads;
    Matrix* concat = createMatrix(batch_size, concat_cols); // (batch_size, total_head_dim)

    for (int i = 0; i < mha->num_heads; i++) {
        for (int r = 0; r < heads[i]->rows; r++) {
            for (int c = 0; c < heads[i]->columns; c++) {
                concat->data[r * concat->columns + (i * mha->head_dim + c)] = heads[i]->data[r * heads[i]->columns + c];
            }
        }
        freeMatrix(heads[i]);
    }
    free(heads);

    // Project output back to input_dim
    Matrix* concat_T = transposeMatrix(concat); // (concat_dim, batch_size)
    Matrix* output = multiplyMatrix(mha->Wo->weights, concat_T); // (input_dim, batch_size)
    freeMatrix(concat_T);
    freeMatrix(concat);

    // Add bias
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->columns; j++) {
            output->data[i * output->columns + j] += mha->Wo->bias->data[i];
        }
    }

    Matrix* output_T = transposeMatrix(output);  // (seq_len × head_dim)
    freeMatrix(output);

    return output_T;
}

void freeMultiHeadAttention(MultiHeadAttention* mha) {
    for (int i = 0; i < mha->num_heads; i++) {
        freeLinearLayer(mha->Wq[i]);
        freeLinearLayer(mha->Wk[i]);
        freeLinearLayer(mha->Wv[i]);
    }
    free(mha->Wq);
    free(mha->Wk);
    free(mha->Wv);
    freeLinearLayer(mha->Wo);
    free(mha);
}
