all: cpu gpu

cpu: main.c
	gcc -o cpu main.c -lm

gpu: gpu.o
	nvcc -o gpu gpu.o

gpu.o: gpu.cu
	nvcc -c -o gpu.o gpu.cu