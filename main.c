#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/GPT.h"

int main() {
    const char* vocab[49] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
        "hello", "world", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "[PAD]", "[UNK]", "[SEP]", "[CLS]", ".", ",", "!", "?", "-", "+", "*", "(", ")"
    };

    char input[1024];
    printf("Enter your input: ");
    fgets(input, sizeof(input), stdin);

    size_t len = strlen(input);
    if (len > 0 && input[len - 1] == '\n') {
        input[len - 1] = '\0';
    }

    char* output = run_transformer_model(vocab, input, 768, 768, 96);

    if (output) {
        printf("Model output: %s\n", output);
        free(output);
    } else {
        printf("Model run failed.\n");
    }

    return 0;
}
