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
    int seq_len = input->columns;       // input: [input_dim][seq_len]
    int head_dim = mha->head_dim;
    int num_heads = mha->num_heads;
    int input_dim = mha->input_dim;

    // Allocate space for attention heads
    Matrix** heads = (Matrix**)malloc(num_heads * sizeof(Matrix*));

    // Compute Q, K, V and attention for each head
    for (int i = 0; i < num_heads; i++) {
        Matrix* Q = multiplyMatrix(mha->Wq[i]->weights, input);  // [head_dim][seq_len]
        Matrix* K = multiplyMatrix(mha->Wk[i]->weights, input);  // [head_dim][seq_len]
        Matrix* V = multiplyMatrix(mha->Wv[i]->weights, input);  // [head_dim][seq_len]

        heads[i] = attentionHead(Q, K, V);  // Output: [head_dim][seq_len]

        freeMatrix(Q);
        freeMatrix(K);
        freeMatrix(V);
    }

    // Concatenate heads into a single matrix: [input_dim][seq_len]
    Matrix* concat = createMatrix(input_dim, seq_len);  // [24][seq_len] if dim=24

    for (int head_idx = 0; head_idx < num_heads; head_idx++) {
        Matrix* attention_head = heads[head_idx];  // [head_dim][seq_len]

        if (!attention_head) {
            printf("Error: attention_head[%d] is NULL!\n", head_idx);
            exit(1);
        }

        for (int token_idx = 0; token_idx < seq_len; token_idx++) { // columns
            for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {  // rows
                int out_row = head_idx * head_dim + dim_idx;
                int out_col = token_idx;

                if (out_row >= concat->rows || out_col >= concat->columns) {
                    printf("Index out of bounds: out_row=%d, out_col=%d\n", out_row, out_col);
                    exit(1);
                }

                // column-major: data[col * rows + row]
                concat->data[out_col * concat->rows + out_row] =
                    attention_head->data[token_idx * attention_head->rows + dim_idx];
            }
        }

        freeMatrix(attention_head);
    }

    // Project back to input_dim using output projection: Wo * concat
    Matrix* output = multiplyMatrix(mha->Wo->weights, concat);  // [input_dim][seq_len]

    // Add bias
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->columns; j++) {
            output->data[j * output->rows + i] += mha->Wo->bias->data[i];
        }
    }

    freeMatrix(concat);
    free(heads);

    return output;  // [input_dim][seq_len]
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
