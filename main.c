#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/imports.h"

int main() {
    // Start the clock before the loop
    clock_t start_time = clock();

    for (int i = 0; i < 1000; i++) {

        // Initialize random seed
        srand((unsigned int)time(NULL));

        // 1. Create input matrix (input_dim = 8, batch_size = 2)
        Matrix* input = createMatrix(8, 2);  // 8 features, 2 examples

        // Fill input with sequential values or random numbers
        for (int i = 0; i < input->rows * input->columns; i++) {
            input->data[i] = (float)(i + 1);
        }

        // 2. Create a Multi-Head Attention block (input_dim = 8, num_heads = 2)
        MultiHeadAttention* mha = createMultiHeadAttention(8, 2);  // 2 heads

        // 3. Forward pass (Apply multi-head attention)
        Matrix* output = multiHeadAttentionForward(mha, input);

        // 4. Free memory
        freeMatrix(input);
        freeMatrix(output);
        freeMultiHeadAttention(mha);
    }

    // Stop the clock after the loop
    clock_t end_time = clock();

    // Calculate the elapsed time
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nTotal time for 1000 iterations: %f seconds\n", time_taken);

    return 0;
}
