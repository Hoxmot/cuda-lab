all: task1 task11

../tools/handlers.o: ../tools/handlers.cu
	nvcc -c -o ../tools/handlers.o ../tools/handlers.cu

task1.o: task1.cu ../tools/handlers.h
	nvcc -c -o task1.o task1.cu

task1: task1.o ../tools/handlers.o
	nvcc -o task1 task1.o ../tools/handlers.o

task11: task1_1.cu
	nvcc -o task11 task1_1.cu

clean:
	rm -f *.o task1

.PHONY: all clean