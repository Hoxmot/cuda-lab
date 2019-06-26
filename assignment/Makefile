all: cpu gpu

cpu: main.o
	gcc -o cpu main.o -lm

main.o: main.c
	gcc -c -o main.o main.c

gpu: gpu.o
	nvcc -o gpu gpu.o

gpu.o: gpu.cu
	nvcc -c -o gpu.o gpu.cu

.PHONY: clean all

clean:
	rm -f cpu gpu gpu.o main.o