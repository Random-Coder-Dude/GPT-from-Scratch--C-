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
    for (int row = 0; row < matrix->rows; row++) {
        for (int column = 0; column < matrix->columns; column++) {
            printf("%8.3f ", matrix->data[row * matrix->columns + column]);
        }
        printf("\n");
    }
}

float getValue(Matrix* matrix, int row, int column) {
    return matrix->data[row * matrix->columns + column];
}

void setValue(Matrix* matrix, int row, int column, float value) {
    matrix->data[row * matrix->columns + column] = value;
}

Matrix* multiplyMatrix(Matrix* a, Matrix* b) {
    if (a->columns != b->rows) {
        printf("Size mismatch %d vs %d\n", a->columns, b->rows);
        return NULL;
    }

    Matrix* product = createMatrix(a->rows, b->columns);

    for (int rowA = 0; rowA < a->rows; rowA++) {
        for (int columnB = 0; columnB < b->columns; columnB++) {
            float sum = 0.0f;
            for (int columnA = 0; columnA < a->columns; columnA++) {
                sum += getValue(a, rowA, columnA) * getValue(b, columnA, columnB);
            }
            setValue(product, rowA, columnB, sum);
        }
    }

    return product;
}

Matrix* addMatrix(Matrix* a, Matrix* b) {
    if (a->columns != b->columns || a->rows != b->rows) {
        printf("Size mismatch %dx%d vs %dx%d\n", a->columns, a->rows, b->columns, b->rows);
        return NULL;
    }
    
    Matrix* sum = createMatrix(a->rows, a->columns);

    for (int row = 0; row < a->rows; row++) {
        for (int column = 0; column < a->columns; column++) {
            float value = getValue(a, row, column) + getValue(b, row, column);
            setValue(sum, row, column, value);
        }
    }

    return sum;
}

Matrix* scalarMultiplyMatrix(float scalar, Matrix* a) {
    Matrix* scaled = createMatrix(a->rows, a->columns);

    for (int row = 0; row < a->rows; row++) {
        for (int column = 0; column < a->columns; column++) {
            setValue(scaled, row, column, scalar * getValue(a, row, column));
        }
    }

    return scaled;
}