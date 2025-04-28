CC=gcc
CFLAGS=-Wall
TARGET=builds/main

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $(TARGET) main.c src/matrixUtils.c src/normalization.c

clean:
	rm -f $(TARGET)
	clear

run: $(TARGET)
	./$(TARGET).exe
