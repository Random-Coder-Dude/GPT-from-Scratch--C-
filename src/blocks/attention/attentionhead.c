#include <math.h>
#include "matrixUtils.h"
#include "softmax.h"
#include "attentionHead.h"

Matrix* attentionHead(Matrix* Q, Matrix* K, Matrix* V) {
    // 1. Transpose K
    Matrix* K_T = transposeMatrix(K);

    // 2. Compute Q × Kᵀ
    Matrix* scores = multiplyMatrix(Q, K_T);

    // 3. Scale scores by sqrt(d_k)
    int d_k = K->columns;
    float scale = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < scores->rows * scores->columns; i++) {
        scores->data[i] *= scale;
    }

    // 4. Apply softmax to scores
    softmaxMatrix(scores);

    // 5. Multiply scores × V
    Matrix* output = multiplyMatrix(scores, V);

    // 6. Clean up temporary matrices
    freeMatrix(K_T);
    freeMatrix(scores);

    return output; // output is the attention result
}
