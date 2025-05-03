#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int tokenize(const char *text, int *token_ids) {
    int index = 0;
    const char *ptr = text;
    
    while (*ptr != '\0') {
        // Skip leading spaces
        while (isspace(*ptr)) ptr++;

        if (*ptr == '\0') break;  // End of string

        const char *start = ptr;
        
        // Move ptr to the end of the current word
        while (*ptr != '\0' && !isspace(*ptr)) ptr++;

        int len = ptr - start;
        if (len > 0) {
            token_ids[index++] = len;  // Token ID = length of token
        }
    }

    return index;  // Return number of tokens
}
