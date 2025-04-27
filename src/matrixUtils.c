#include <stdio.h>
#include <stdlib.h>
#include "matrixUtils.h"

//Create a new Matrix initialized with all 0's
Matrix* createMatrix(int rows, int columns) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->data = (float*)calloc(rows*columns, sizeof(float));
    return matrix;
}

//Delete the Matrix and prevent memory leaks
void freeMatrix(Matrix* matrix) {
    if (matrix != NULL) {
        free(matrix->data);
        free(matrix);
    }
}

//Print the Matrix into the terminal
void printMatrix(Matrix* matrix) {
    if (matrix == NULL) {
        printf("Matrix is NULL\n");
        return;
    }
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            printf("%8.3f ", matrix->data[i * matrix->columns + j]);
        }
        printf("\n");
    }
}

void multiplyMatrix(Matrix* a, Matrix* b) {
    
}