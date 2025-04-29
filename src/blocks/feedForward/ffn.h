#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "matrixUtils.h"
#include "linearLayer.h"

typedef struct {
    LinearLayer* fc1;
    LinearLayer* fc2;
} FeedForward;

FeedForward* createFeedForward(int input_dim, int hidden_dim);
Matrix* feedForwardForward(FeedForward* ffn, Matrix* input);
void freeFeedForward(FeedForward* ffn);

#endif
