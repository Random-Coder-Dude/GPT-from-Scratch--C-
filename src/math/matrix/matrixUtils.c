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
        printf("matrix doesn't exist\n");
        exit(1);
    }
    for (int row = 0; row < matrix->rows; row++) {
        for (int column = 0; column < matrix->columns; column++) {
            printf("%8.3f ", matrix->data[row * matrix->columns + column]);
        }
        printf("\n");
    }
}

//Get the specified value from a matrix
float getValue(Matrix* matrix, int row, int column) {
    return matrix->data[row * matrix->columns + column];
}

//Set the specifified value into a matrix
void setValue(Matrix* matrix, int row, int column, float value) {
    matrix->data[row * matrix->columns + column] = value;
}

//Multiply 2 matrixes with each other
Matrix* multiplyMatrix(Matrix* a, Matrix* b) {
    if (a->columns != b->rows) {
        printf("Size Mismatch: %dx%d\n", a->columns, b->rows);
        exit(1);
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

//Add 2 matrixes together
Matrix* addMatrix(Matrix* a, Matrix* b) {
    if (a->columns != b->columns || a->rows != b->rows) {
        printf("mismatch matrix dimesnions (ADD MATRIX): %dvs%d, %dvs%d\n", a->rows, b->rows, a->columns, b->columns);
        exit(1);
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

//Multiply a matrix by a scalar
Matrix* scalarMultiplyMatrix(float scalar, Matrix* a) {
    Matrix* scaled = createMatrix(a->rows, a->columns);

    for (int row = 0; row < a->rows; row++) {
        for (int column = 0; column < a->columns; column++) {
            setValue(scaled, row, column, scalar * getValue(a, row, column));
        }
    }

    return scaled;
}

//Flips the rows and columns on a matrix
Matrix* transposeMatrix(Matrix* matrix) {
    Matrix* result = createMatrix(matrix->columns, matrix->rows);

    for (int row = 0; row < matrix->rows; row++) {
        for (int column = 0; column < matrix->columns; column++) {
            result->data[column * result->columns + row] = matrix->data[row * matrix->columns + column];
        }
    }

    return result;
}

int argmaxRow(Matrix* logits, int row_index) {
    int vocab_size = logits->columns;
    float max_val = logits->data[row_index * vocab_size];
    int max_index = 0;

    for (int i = 1; i < vocab_size; i++) {
        float val = logits->data[row_index * vocab_size + i];
        if (val > max_val) {
            max_val = val;
            max_index = i;
        }
    }

    return max_index;
}