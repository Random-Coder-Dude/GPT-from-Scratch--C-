#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "matrixUtils.h"

// Generate (max_len × dim) matrix of positional encodings
Matrix* createPositionalEncoding(int max_len, int dim);

#endif
