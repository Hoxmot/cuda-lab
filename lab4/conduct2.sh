#!/bin/bash

echo "Conducting tests for exercise 2"

echo "Test 1/4"
echo "" >result2_1
rm result2_1
touch result2_1
for i in 10 20 50 100 200 500 1000 5000 10000 50000 100000 500000 1000000 5000000 10000000 50000000 100000000 500000000; do
    echo "Len: $i"
    echo $i >>result2_1
    echo "" >>result2_1
    nvcc -DLEN=$i -DBLOCKS=2 -DTHREADS=10 -o task2_1 task2_1.cu
    nvcc -DLEN=$i -DBLOCKS=2 -DTHREADS=10 -o task2_2 task2_2.cu
    time ./task2_1 >>result2_1
    time ./task2_2 >>result2_1
    echo "" >>result2_1
    echo "--------------------------------------------------" >>result2_1
    echo "" >>result2_1
done

echo "Test 2/4"
echo "" >result2_2
rm result2_2
touch result2_2
for i in 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512; do
    echo "Threads per block: $i"
    echo $i >>result2_2
    echo "" >>result2_2
    nvcc -DLEN=100000000 -DBLOCKS=2 -DTHREADS=$i -o task2_1 task2_1.cu
    nvcc -DLEN=100000000 -DBLOCKS=2 -DTHREADS=$i -o task2_2 task2_2.cu
    time ./task2_1 >>result2_2
    time ./task2_2 >>result2_2
    echo "" >>result2_2
    echo "--------------------------------------------------" >>result2_2
    echo "" >>result2_2
done

echo "Test 3/4"
echo "" >result2_3
rm result2_3
touch result2_3
for i in 30 60 100 130 190 220 260 300 350 380 420 450 510; do
    echo "Threads per block: $i"
    echo $i >>result2_3
    echo "" >>result2_3
    nvcc -DLEN=100000000 -DBLOCKS=2 -DTHREADS=$i -o task2_1 task2_1.cu
    nvcc -DLEN=100000000 -DBLOCKS=2 -DTHREADS=$i -o task2_2 task2_2.cu
    time ./task2_1 >>result2_3
    time ./task2_2 >>result2_3
    echo "" >>result2_3
    echo "--------------------------------------------------" >>result2_3
    echo "" >>result2_3
done

echo "Test 4/4"
echo "" >result2_4
rm result2_4
touch result2_4
for i in 2 4 8 16 32 64 96 128 150 200 256 300 400 512; do
    echo "Number of blocks: $i"
    echo $i >>result2_4
    echo "" >>result2_4
    nvcc -DLEN=100000000 -DBLOCKS=$i -DTHREADS=32 -o task2_1 task2_1.cu
    nvcc -DLEN=100000000 -DBLOCKS=$i -DTHREADS=32 -o task2_2 task2_2.cu
    time ./task2_1 >>result2_4
    time ./task2_2 >>result2_4
    echo "" >>result2_4
    echo "--------------------------------------------------" >>result2_4
    echo "" >>result2_4
done

echo "Tests done"
