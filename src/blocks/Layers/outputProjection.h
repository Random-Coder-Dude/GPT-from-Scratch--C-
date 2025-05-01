#ifndef OUTPUT_PROJECTION_H
#define OUTPUT_PROJECTION_H

#include "matrixUtils.h"
#include "linearLayer.h"

typedef struct {
    LinearLayer* proj;
} OutputProjection;

OutputProjection* createOutputProjection(int input_dim, int vocab_size);
Matrix* outputForward(OutputProjection* out_proj, Matrix* input);
void freeOutputProjection(OutputProjection* out_proj);

#endif
