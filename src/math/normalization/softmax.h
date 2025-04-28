#ifndef NORMALIZATION_H
#define NORMALIZATION_H
#include "matrixUtils.h"

void softmaxRow(Matrix* matrix, int row_index);
void softmaxMatrix(Matrix* matrix);
void layerNormalization(Matrix* matrix, float epsilon);

#endif