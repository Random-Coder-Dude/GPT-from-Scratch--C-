// demo.c

#include <stdio.h>
#include "src/imports.h"

int main() {
    // Sample text to tokenize
    const char* text = "hello world from tokenizer demo";

    // Array to hold the token IDs
    int token_ids[MAX_TOKENS];

    // Tokenize the input text
    int num_tokens = tokenize(text, token_ids);

    // Print the results
    printf("Original text: \"%s\"\n", text);
    printf("Tokenized IDs:\n");

    for (int i = 0; i < num_tokens; i++) {
        printf("Token %d ID: %d\n", i + 1, token_ids[i]);
    }

    return 0;
}
