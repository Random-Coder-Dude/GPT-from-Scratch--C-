#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

typedef struct {
    int rows;
    int columns;
    float* data;
} Matrix;

// Matrix Operations
Matrix* createMatrix(int rows, int columns);
void freeMatrix(Matrix* matrix);
void printMatrix(Matrix* matrix);

#endif