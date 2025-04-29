#include <math.h>
#include "normalization.h"
#include "matrixUtils.h"

void layerNormalization(Matrix* matrix, float epsilon) {
    for (int m_rows = 0; m_rows < matrix->rows; m_rows++) {
        int columns = matrix->columns;
        float sum = 0.0f;

        for (int m_columns = 0; m_columns < columns; m_columns++) {
            sum += matrix->data[m_rows * columns + m_columns];
        }

        float mean = sum / columns;
        float variance = 0.0f;

        for (int m_columns = 0; m_columns < columns; m_columns++) {
            float diff = matrix->data[m_rows * columns + m_columns] - mean;
            variance += diff * diff;
        }

        variance /= columns;

        for (int m_columns = 0; m_columns < columns; m_columns++) {
            matrix->data[m_rows * columns + m_columns] = (matrix->data[m_rows * columns + m_columns] - mean) / sqrtf(variance + epsilon);
        }
    }
}
