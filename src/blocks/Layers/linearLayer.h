#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "matrixUtils.h"

typedef struct {
    Matrix* weights;
    Matrix* bias;
} LinearLayer;

LinearLayer* createLinearLayer(int input_dim, int output_dim);
Matrix* linearForward(LinearLayer* layer, Matrix* input);
void freeLinearLayer(LinearLayer* layer);

#endif
