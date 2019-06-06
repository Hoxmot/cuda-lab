//
//  Mandelbrot.cu
//  
//
//  Created by Witold Rudnicki on 22.05.2015.
//
//

#include <cuda_runtime_api.h>
#include <math.h>

#include "Mandelbrot.h"

#define BLOCK_SIZE 32

__global__ void cudaMandel(double* X0, double* Y0, double* X1, double* Y1, int* POZ, int* PION, int* ITER,int* Mandel);

void handleCudaMalloc(void **var, ssize_t size) {
    cudaError_t status;
    status = cudaMalloc(var, size);
    if (status != cudaSuccess) {
	    printf("%s\n", cudaGetErrorString(status));
    }
}

void handleCudaMemcpy(void* dst, const void* src, ssize_t size, cudaMemcpyKind kind) {
    cudaError_t status;
    status = cudaMemcpy(dst, src, size, kind);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }
}

void handleCudaFree(void* pointer) {
    cudaError_t status;
    status = cudaFree(pointer);
    if (status != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(status));
    }
}

int main(int argc, char **argv) {

#ifdef PARAM
    //Ustaw obszar obliczeń
    //{X0,Y0} - lewy dolny róg
    double X0=atof(argv[1]);
    double Y0=atof(argv[2]);

    //{X1,Y1} - prawy górny róg
    double X1=atof(argv[3]);
    double Y1=atof(argv[4]);

    //Ustal rozmiar w pikselach
    //{POZ,PION}
    int POZ=atoi(argv[5]);
    int PION=atoi(argv[6]);

    //Ustal liczbę iteracji próbkowania
    //{ITER}

    int ITER=atoi(argv[7]);
#else
    double X0 = -1.5;
    double Y0 = -0.5;
    double X1 = -1.1;
    double Y1 = -0.1;
    int POZ = 2048;
    int PION = 2048;
    int ITER = 2566;
#endif /* PARAM */

    printf("X0 = %f\n", X0);
    printf("Y0 = %f\n", Y0);
    printf("X1 = %f\n", X1);
    printf("Y1 = %f\n", Y1);
    printf("POZ = %d\n", POZ);
    printf("PION = %d\n", PION);
    printf("ITER = %d\n", ITER);

    //Zaalokuj tablicę do przechowywania wyniku
    int *Result;
    Result= (int*) malloc(sizeof(int)*POZ*PION);

    int *Result_gpu;
    double *x0_gpu, *y0_gpu, *x1_gpu, *y1_gpu;
    int *poz_gpu, *pion_gpu, *iter_gpu;

    // Result
    handleCudaMalloc((void**)&Result_gpu, sizeof(int) * POZ * PION);

    // X0
    handleCudaMalloc((void**)&x0_gpu, sizeof(double));
    handleCudaMemcpy(x0_gpu, &X0, sizeof(double), cudaMemcpyHostToDevice);

    //  Y0
    handleCudaMalloc((void**)&y0_gpu, sizeof(double));
    handleCudaMemcpy(y0_gpu, &Y0, sizeof(double), cudaMemcpyHostToDevice);

    // X1
    handleCudaMalloc((void**)&x1_gpu, sizeof(double));
    handleCudaMemcpy(x1_gpu, &X1, sizeof(double), cudaMemcpyHostToDevice);

    // Y1
    handleCudaMalloc((void**)&y1_gpu, sizeof(double));
    handleCudaMemcpy(y1_gpu, &Y1, sizeof(double), cudaMemcpyHostToDevice);

    // POZ
    handleCudaMalloc((void**)&poz_gpu, sizeof(int));
    handleCudaMemcpy(poz_gpu, &POZ, sizeof(int), cudaMemcpyHostToDevice);

    // PION
    handleCudaMalloc((void**)&pion_gpu, sizeof(int));
    handleCudaMemcpy(pion_gpu, &PION, sizeof(int), cudaMemcpyHostToDevice);

    // ITER
    handleCudaMalloc((void**)&iter_gpu, sizeof(int));
    handleCudaMemcpy(iter_gpu, &ITER, sizeof(int), cudaMemcpyHostToDevice);
    
    time_t start, end;
    // do computations

    printf("Computations for rectangle { (%lf %lf), (%lf %lf) }\n",X0,Y0,X1,Y1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(POZ / dimBlock.x, PION / dimBlock.y);
    start=clock();
    cudaMandel<<<dimGrid, dimBlock>>>(x0_gpu, y0_gpu, x1_gpu, y1_gpu, poz_gpu, pion_gpu, iter_gpu, Result_gpu);
    cudaDeviceSynchronize();
    end=clock();
    
    printf("\nComputing took %lf s\n\n", 1.0 * (end-start) / CLOCKS_PER_SEC);
    
    handleCudaMemcpy(Result, Result_gpu, sizeof(int) * POZ * PION, cudaMemcpyDeviceToHost);

    start=clock();
    makePicture(Result, POZ, PION, ITER);
    end=clock();
    printf("Saving took %lf s\n\n", 1.0 * (end-start) / CLOCKS_PER_SEC);

    handleCudaFree(x0_gpu);
    handleCudaFree(y0_gpu);
    handleCudaFree(x1_gpu);
    handleCudaFree(y1_gpu);
    handleCudaFree(poz_gpu);
    handleCudaFree(pion_gpu);
    handleCudaFree(iter_gpu);
    handleCudaFree(Result_gpu);

    return 0;
}

__global__ void cudaMandel(double* X0, double* Y0, double* X1, double* Y1, int* POZ, int* PION, int* ITER, int* Mandel) {
    
    double dX = (*X1 - *X0) / (*POZ - 1);
    double dY = (*Y1 - *Y0) / (*PION - 1);
    double tZx, tZy;
    int i;

    double x, y, Zx, Zy;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

//    __shared__ double Zx[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ double Zy[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ double x[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__ double y[BLOCK_SIZE][BLOCK_SIZE];

    int poz = blockRow * (*POZ)  + row;
    int pion = blockCol * (*PION) + col;
    if (poz < *POZ && pion < *PION) {
        x = (*X0) + poz * dX;
        y = (*Y0) + pion * dY;
        Zx = x;
        Zy = y;
        i = 0;
        while ((i < *ITER) &&
                ((Zx * Zx + Zy * Zy) < 4) ) {
            
            if (i == 0) {
                printf("%d x %d ; %d x %d\n", blockRow, blockCol, row, col);
            }
            tZx = Zx * Zx - Zy * Zy + x;
            tZy = 2 * Zx * Zy + y;
            Zx = tZx;
            Zy = tZy;
            i++;
        }
        Mandel[pion * (*POZ) + poz] = i;
    }
}

void makePicture(int *Mandel,int width, int height, int MAX) {
    
    int red_value, green_value, blue_value;
    
    int MyPalette[41][3]={
        {255,255,255}, //0
        {255,255,255}, //1 not used
        {255,255,255}, //2 not used
        {255,255,255}, //3 not used
        {255,255,255}, //4 not used
        {255,180,255}, //5
        {255,180,255}, //6 not used
        {255,180,255}, //7 not used
        {248,128,240}, //8
        {248,128,240}, //9 not used
        {240,64,224}, //10
        {240,64,224}, //11 not used
        {232,32,208}, //12
        {224,16,192}, //13
        {216,8,176}, //14
        {208,4,160}, //15
        {200,2,144}, //16
        {192,1,128}, //17
        {184,0,112}, //18
        {176,0,96}, //19
        {168,0,80}, //20
        {160,0,64}, //21
        {152,0,48}, //22
        {144,0,32}, //23
        {136,0,16}, //24
        {128,0,0}, //25
        {120,16,0}, //26
        {112,32,0}, //27
        {104,48,0}, //28
        {96,64,0}, //29
        {88,80,0}, //30
        {80,96,0}, //31
        {72,112,0}, //32
        {64,128,0}, //33
        {56,144,0}, //34
        {48,160,0}, //35
        {40,176,0}, //36
        {32,192,0}, //37
        {16,224,0}, //38
        {8,240,0}, //39
        {0,0,0} //40
    };
    
    FILE *f = fopen("cudaMandel.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", width, height);
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            // compute index to the palette
            int indx= (int) floor(5.0*log2f(1.0f*Mandel[j*width+i]+1));
            red_value=MyPalette[indx][0];
            green_value=MyPalette[indx][2];
            blue_value=MyPalette[indx][1];
            
            fputc(red_value, f);   // 0 .. 255
            fputc(green_value, f); // 0 .. 255
            fputc(blue_value, f);  // 0 .. 255
        }
    }
    fclose(f);
    
}
