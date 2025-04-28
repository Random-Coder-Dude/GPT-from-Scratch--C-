#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/imports.h"

int main() {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // 1. Create input matrix (input_dim = 8, batch_size = 2)
    Matrix* input = createMatrix(8, 2);  // 8 features, 2 examples

    // Fill input with sequential values or random numbers
    for (int i = 0; i < input->rows * input->columns; i++) {
        input->data[i] = (float)(i + 1);
    }

    printf("Input Matrix:\n");
    printMatrix(input);

    // 2. Create a Multi-Head Attention block (input_dim = 8, num_heads = 2)
    MultiHeadAttention* mha = createMultiHeadAttention(8, 2);  // 2 heads

    // 3. Forward pass
    Matrix* output = multiHeadAttentionForward(mha, input);

    printf("\nOutput after Multi-Head Attention:\n");
    printMatrix(output);

    // 4. Free memory
    freeMatrix(input);
    freeMatrix(output);
    freeMultiHeadAttention(mha);

    return 0;
}
