#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "matrixUtils.h"

// Applies LayerNorm to each row in the matrix (row-wise normalization)
void layerNormalization(Matrix* matrix, float epsilon);

#endif
