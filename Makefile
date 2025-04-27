CC=gcc
CFLAGS=-Wall
TARGET=builds

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $(TARGET) main.c src/matrixUtils.c

clean:
	rm -f $(TARGET)
	clear

run: $(TARGET)
	./$(TARGET).exe
