
.PHONY: all clean

all: task0 task1

task0: task0.c limit.h
	cc -o task0 task0.c limit.h

main.o: main.cu reduce.h limit.h
	nvcc -c -o main.o main.cu reduce.h limit.h

task1.o: task1.cu limit.h
	nvcc -c -o task1.o task1.cu limit.h

task1: task1.o main.o
	nvcc -o task1 main.o task1.o

clean:
	rm -f *.o task0 task1
