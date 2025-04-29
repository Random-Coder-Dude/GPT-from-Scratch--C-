# Compiler and flags
CC          = gcc
CFLAGS      = -Wall -lm
INCLUDES    = -I./src/math/matrix \
             -I./src/math/normalization \
             -I./src/blocks/attention \
             -I./src/blocks/linearLayer \
             -I./src/blocks/feedforward \
             -I./src/blocks/transformer \

# Directories
SRC_DIR     = src
BUILD_DIR   = builds

# Target executable
TARGET      = $(BUILD_DIR)/main

# Source files
FILES       = main.c \
             $(SRC_DIR)/math/matrix/matrixUtils.c \
             $(SRC_DIR)/math/normalization/softmax.c \
             $(SRC_DIR)/blocks/attention/attentionhead.c \
             $(SRC_DIR)/blocks/attention/multiHeadAttention.c \
             $(SRC_DIR)/blocks/linearLayer/linearLayer.c \
             $(SRC_DIR)/math/normalization/normalization.c \
             $(SRC_DIR)/blocks/feedforward/ffn.c \
             $(SRC_DIR)/blocks/transformer/transformerBlock.c

# Default target: build and run the program
run: $(TARGET)
	./$(TARGET)

# Rule to build the target
$(TARGET): $(FILES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(FILES)

# Clean up build files and clear the output
clean:
	rm -f $(TARGET)
	clear
