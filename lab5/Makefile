all: task1

../tools/handlers.o: ../tools/handlers.cu
	nvcc -c -o ../tools/handlers.o ../tools/handlers.cu

task1.o: task1.cu ../tools/handlers.h
	nvcc -DLEN=32 -c -o task1.o task1.cu

task1: task1.o ../tools/handlers.o
	nvcc -o task1 task1.o ../tools/handlers.o

clean:
	rm -f *.o task1

.PHONY: all clean