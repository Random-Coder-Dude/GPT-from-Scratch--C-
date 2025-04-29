#include <stdio.h>
#include "src/imports.h"

int main() {
    int input_dim = 8;
    int batch_size = 2;
    int hidden_dim = 16;
    int num_heads = 2;

    Matrix* input = createMatrix(input_dim, batch_size);
    for (int i = 0; i < input->rows * input->columns; i++) {
        input->data[i] = (float)(i + 1);
    }

    TransformerBlock* block = createTransformerBlock(input_dim, num_heads, hidden_dim);

    printf("Input:\n");
    printMatrix(input);

    Matrix* output = transformerForward(block, input);

    printf("\nOutput after Transformer Block:\n");
    printMatrix(output);

    freeMatrix(input);
    freeMatrix(output);
    freeTransformerBlock(block);
    
    return 0;
}
