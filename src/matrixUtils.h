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
float getValue(Matrix* matrix, int row, int column);
void setValue(Matrix* matrix, int row, int column, float value);
Matrix* multiplyMatrix(Matrix* a, Matrix* b);
Matrix* addMatrix(Matrix* a, Matrix* b);
Matrix* scalarMultiplyMatrix(float scalar, Matrix* a);

#endif