#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

// Function to tokenize input text and convert it to token IDs
int tokenize(const char *text, int *token_ids) {
    const char *delim = " ";  // Tokenize based on spaces
    char *text_copy = strdup(text);  // Make a copy of the input string to split
    char *token = strtok(text_copy, delim);
    int index = 0;

    // Split the string into tokens and assign each token a unique ID (here, length of token)
    while (token != NULL) {
        token_ids[index++] = strlen(token);  // In a real case, use a vocabulary for mapping
        token = strtok(NULL, delim);
    }

    free(text_copy);  // Free the copied string
    return index;  // Return the number of tokens
}
