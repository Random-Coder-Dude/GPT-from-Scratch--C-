#include <stdio.h>
#include "src/transformer.h"

int main() {
    // 1. Create a matrix (2x3)
    Matrix* matrix = createMatrix(2, 3);

    // 2. Fill the matrix with some values
    matrix->data[0] = 2.0f;
    matrix->data[1] = 2.0f;
    matrix->data[2] = 3.0f;
    matrix->data[3] = 5.0f;
    matrix->data[4] = 5.0f;
    matrix->data[5] = 6.0f;

    printf("Original matrix:\n");
    printMatrix(matrix);

    // 3. Apply Layer Normalization row-wise
    layerNormalization(matrix, 1e-5f);  // epsilon = 1e-5 to avoid division by zero

    printf("\nAfter applying Layer Normalization:\n");
    printMatrix(matrix);

    // 4. Free memory
    freeMatrix(matrix);

    return 0;
}
