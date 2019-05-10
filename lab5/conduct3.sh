#!/bin/bash

echo "Conducting tests for exercise 3"

echo "Test 1/3"
echo "" >result3_1
rm result3_1
touch result3_1
for i in 32 64 96 128 256 1024 2048 4096 8192 10240 ; do
    echo "shared cuda, len: $i"
    echo $i >>result3_1
    nvcc -DLEN=$i -o task1 task1.cu
    { time ./task1; } 2>>result3_1
    echo "--------------------------------------------------" >>result3_1
done

echo "Test 2/3"
echo "" >result3_2
rm result3_2
touch result3_2
for i in 32 64 96 128 256 1024 2048 4096 8192 10240 ; do
    echo "cuda, len: $i"
    echo $i >>result3_2
    nvcc -DLEN=$i -o task2 task2.cu
    { time ./task2; } 2>>result3_2
    echo "--------------------------------------------------" >>result3_2
done

echo "Test 3/3"
echo "" >result3_3
rm result3_3
touch result3_3
for i in 32 64 96 128 256 1024 2048 4096 8192 10240 ; do
    echo "cpu, len: $i"
    echo $i >>result3_3
    gcc -std=c99 -O2 -Wall -Wextra -DLEN=$i -o task3 task3.c
    { time ./task3; } 2>>result3_3
    echo "--------------------------------------------------" >>result3_3
done