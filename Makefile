# Compiler and flags
CC        = gcc
CFLAGS    = -Wall
LDFLAGS   = -lm
INCLUDES  = -I./src/math/matrix \
            -I./src/math/normalization \
            -I./src/blocks/attention \
            -I./src/blocks/layers \
            -I./src/blocks/feedforward \
            -I./src/blocks/transformer \
            -I./src/blocks/tokenizer

# Directories
SRC_DIR   = src
BUILD_DIR = builds
TARGET    = $(BUILD_DIR)/main

FILES     = main.c \
            $(SRC_DIR)/math/matrix/matrixUtils.c \
            $(SRC_DIR)/math/normalization/softmax.c \
            $(SRC_DIR)/blocks/attention/attentionhead.c \
            $(SRC_DIR)/blocks/attention/multiHeadAttention.c \
            $(SRC_DIR)/blocks/layers/linearLayer.c \
            $(SRC_DIR)/math/normalization/normalization.c \
            $(SRC_DIR)/blocks/feedforward/ffn.c \
            $(SRC_DIR)/blocks/transformer/transformerBlock.c \
            $(SRC_DIR)/blocks/tokenizer/tokenizer.c \
            $(SRC_DIR)/blocks/layers/embedding.c \
            $(SRC_DIR)/blocks/tokenizer/positionalEncoding.c \
            $(SRC_DIR)/blocks/layers/outputProjection.c

.PHONY: run clean

run: $(TARGET)
	-./$(TARGET) || true

$(TARGET): $(FILES)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(FILES) $(LDFLAGS)

clean:
	rm -f $(TARGET)
	clear
