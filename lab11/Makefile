

all: main cuda_main

main: main.o
	gcc -lm -o main main.o

main.o: Mandelbrot.cpp Mandelbrot.h
	gcc -c -o main.o Mandelbrot.cpp

cuda_main: cuda_main.o
	nvcc -o cuda_main cuda_main.o

cuda_main.o: Mandelbrot.cu Mandelbrot.h
	nvcc -c -o cuda_main.o Mandelbrot.cu

clean:
	rm -f *.o main cuda_main *.ppm
