#include <math.h>
#include "positionalEncoding.h"
#include "matrixUtils.h"

Matrix* createPositionalEncoding(int max_len, int dim) {
    Matrix* encoding = createMatrix(max_len, dim);

    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < dim; i++) {
            float angle = pos / powf(10000.0f, (2 * (i / 2)) / (float)dim);

            float value = (i % 2 == 0)
                ? sinf(angle)
                : cosf(angle);

            setValue(encoding, pos, i, value);
        }
    }

    return encoding;
}
