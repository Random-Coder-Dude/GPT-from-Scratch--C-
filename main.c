#include <stdio.h>
#include "src/matrixUtils.h"

int main() {
    // 1. Create a 2x3 matrix
    Matrix* mat = createMatrix(2, 3);

    // 2. Set values manually
    mat->data[0] = 1.1f;  // Row 0, Col 0
    mat->data[1] = 2.2f;  // Row 0, Col 1
    mat->data[2] = 3.3f;  // Row 0, Col 2
    mat->data[3] = 4.4f;  // Row 1, Col 0
    mat->data[4] = 5.5f;  // Row 1, Col 1
    mat->data[5] = 6.6f;  // Row 1, Col 2

    // 3. Print the matrix
    printf("Matrix contents:\n");
    printMatrix(mat);

    // 4. Free the matrix
    freeMatrix(mat);

    return 0;
}
