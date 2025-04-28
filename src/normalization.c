#include <math.h>
#include "normalization.h"
#include "matrixUtils.h"

//Softmax a certain row in a matrix (Helper Function)
void softmaxRow(Matrix* matrix, int row_index) {
    int columns = matrix->columns;
    float max_val = matrix->data[row_index * columns];

    for (int m_columns = 1; m_columns < columns; m_columns++) {
        float val = matrix->data[row_index * columns + m_columns];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int m_columns = 0; m_columns < columns; m_columns++) {
        matrix->data[row_index * columns + m_columns] = expf(matrix->data[row_index * columns + m_columns] - max_val);
        sum += matrix->data[row_index * columns + m_columns];
    }

    for (int m_columns = 0; m_columns < columns; m_columns++) {
        matrix->data[row_index * columns + m_columns] /= sum;
    }
}

//Softmax an entire matrix in 1 call
void softmaxMatrix(Matrix* matrix) {
    for (int m_rows = 0; m_rows < matrix->rows; m_rows++) {
        softmaxRow(matrix, m_rows);
    }
}

//Normalize a matrix (Layer Normalization)
void layerNormalization(Matrix* matrix, float epsilon) {
    for (int m_rows = 0; m_rows < matrix->rows; m_rows++) {
        int m_columns = matrix->columns;
        float sum = 0.0f;

        for (int m_cols = 0; m_cols < m_columns; m_cols++) {
            sum += matrix->data[m_rows * m_columns + m_cols];
        }

        float mean = sum / m_columns;
        float variance = 0.0f;

        for (int m_cols = 0; m_cols < m_columns; m_cols++) {
            float diff = matrix->data[m_rows * m_columns + m_cols] - mean;
            variance += diff * diff;
        }

        variance /= m_columns;

        for (int m_cols = 0; m_cols < m_columns; m_cols++) {
            matrix->data[m_rows * m_columns + m_cols] = (matrix->data[m_rows * m_columns + m_cols] - mean) / sqrtf(variance + epsilon);
        }
    }
}

