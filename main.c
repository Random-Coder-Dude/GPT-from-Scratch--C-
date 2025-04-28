#include <stdio.h>
#include "src/matrixUtils.h"  // Assuming you have createMatrix, setValue, printMatrix, etc.

int main() {
    // 1. Create a 2x3 and a 3x4 matrix
    Matrix* matrix1 = createMatrix(2, 3);
    Matrix* matrix2 = createMatrix(2, 3);

    // 2. Set values for matrix1 (2x3)
    setValue(matrix1, 0, 0, 1); setValue(matrix1, 0, 1, 2); setValue(matrix1, 0, 2, 3);
    setValue(matrix1, 1, 0, 4); setValue(matrix1, 1, 1, 5); setValue(matrix1, 1, 2, 6);

    // Set values for matrix2 (3x4)
    setValue(matrix2, 0, 0, 7);  setValue(matrix2, 0, 1, 8);  setValue(matrix2, 0, 2, 9);
    setValue(matrix2, 1, 0, 11); setValue(matrix2, 1, 1, 12); setValue(matrix2, 1, 2, 13);

    // 3. Multiply the matrices
    Matrix* result = scalarMultiplyMatrix(2, matrix1);

    // 4. Print the result matrix
    printf("Result of matrix1 + matrix2:\n");
    printMatrix(result);

    // 5. Free memory
    freeMatrix(matrix1);
    freeMatrix(matrix2);
    freeMatrix(result);

    return 0;
}
