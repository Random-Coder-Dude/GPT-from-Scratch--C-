#include <stdlib.h>
#include <stdio.h>
#include "matrixUtils.h"
#include "linearLayer.h"

LinearLayer* createLinearLayer(int input_dim, int output_dim) {
    LinearLayer* layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    layer->weights = createMatrix(output_dim, input_dim);
    layer->bias = createMatrix(output_dim, 1);

    for (int i = 0; i < output_dim * input_dim; i++) {
        layer->weights->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    for (int i = 0; i < output_dim; i++) {
        layer->bias->data[i] = 0.0f;
    }

    return layer;
}

Matrix* linearForward(LinearLayer* layer, Matrix* input) {
    Matrix* temp = multiplyMatrix(layer->weights, input); // (output_dim, batch_size)
    Matrix* output = transposeMatrix(temp); // (batch_size, output_dim)
    freeMatrix(temp);

    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->columns; j++) {
            output->data[i * output->columns + j] += layer->bias->data[j];
        }
    }
    return output;
}



void freeLinearLayer(LinearLayer* layer) {
    freeMatrix(layer->weights);
    freeMatrix(layer->bias);
    free(layer);
}
