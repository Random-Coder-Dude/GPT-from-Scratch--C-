#include <stdio.h>
#include <stdlib.h>
#include "multiHeadAttention.h"
#include "attentionhead.h"

//Create and Allocate a Multi Attention Head
MultiHeadAttention* createMultiHeadAttention(int input_dim, int num_heads) {
    // 1. Allocate space
    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));

    // 2. Add metadata
    mha->num_heads = num_heads;
    mha->input_dim = input_dim;
    mha->head_dim = input_dim / num_heads;

    // 3. Allocate space for Linear Layers
    mha->Wq = (LinearLayer**)malloc(num_heads * sizeof(LinearLayer*));
    mha->Wk = (LinearLayer**)malloc(num_heads * sizeof(LinearLayer*));
    mha->Wv = (LinearLayer**)malloc(num_heads * sizeof(LinearLayer*));

    // 4. Create the Linear Layers for each head
    for (int i = 0; i < num_heads; i++) {
        mha->Wq[i] = createLinearLayer(input_dim, mha->head_dim);
        mha->Wk[i] = createLinearLayer(input_dim, mha->head_dim);
        mha->Wv[i] = createLinearLayer(input_dim, mha->head_dim);
    }

    // 5. Create final Linear Layer
    mha->Wo = createLinearLayer(input_dim, input_dim);

    return mha;
}

//Calculate the full Multi Head Attention pass
Matrix* multiHeadAttentionForward(MultiHeadAttention* mha, Matrix* input) {
    // 1. Set Metadata Variables
    int seq_len = input->columns;
    int head_dim = mha->head_dim;
    int num_heads = mha->num_heads;
    int input_dim = mha->input_dim;

    // 2. Allocate space for attention heads
    Matrix** heads = (Matrix**)malloc(num_heads * sizeof(Matrix*));

    // 3. Compute Q, K, V and attention for each head
    for (int i = 0; i < num_heads; i++) {

        Matrix* Q = multiplyMatrix(mha->Wq[i]->weights, input);
        Matrix* K = multiplyMatrix(mha->Wk[i]->weights, input); 
        Matrix* V = multiplyMatrix(mha->Wv[i]->weights, input);

        heads[i] = attentionHead(Q, K, V);

        freeMatrix(Q);
        freeMatrix(K);
        freeMatrix(V);
    }

    // 4. Concatenate heads into a single matrix
    Matrix* concat = createMatrix(input_dim, seq_len);

    for (int head_idx = 0; head_idx < num_heads; head_idx++) {

        Matrix* attention_head = heads[head_idx];

        for (int token_idx = 0; token_idx < seq_len; token_idx++) {

            for (int dim_idx = 0; dim_idx < head_dim; dim_idx++) {

                int out_row = head_idx * head_dim + dim_idx;
                int out_col = token_idx;
                
                concat->data[out_col * concat->rows + out_row] = attention_head->data[token_idx * attention_head->rows + dim_idx];
            }
        }

        freeMatrix(attention_head);
    }

    // 5. Project back to input_dim using output projection
    Matrix* output = multiplyMatrix(mha->Wo->weights, concat);

    // 6. Add bias
    for (int i = 0; i < output->rows; i++) {

        float b = mha->Wo->bias->data[i];
        for (int j = 0; j < output->columns; j++) {

            output->data[j * output->rows + i] += b;
        }
    }

    // 7. Free Memory
    freeMatrix(concat);
    free(heads);

    return output;
}

//Free the memory of the Multi Head Attention
void freeMultiHeadAttention(MultiHeadAttention* mha) {
    // 1. Free each Linear Layer
    for (int i = 0; i < mha->num_heads; i++) {
        freeLinearLayer(mha->Wq[i]);
        freeLinearLayer(mha->Wk[i]);
        freeLinearLayer(mha->Wv[i]);
    }

    // 2. Free Metadata
    free(mha->Wq);
    free(mha->Wk);
    free(mha->Wv);
    freeLinearLayer(mha->Wo);

    // 3. Free the Multi Head Attention
    free(mha);
}
