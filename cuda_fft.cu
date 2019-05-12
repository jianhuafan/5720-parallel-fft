// compile command: /usr/local/cuda-10.0/bin/nvcc -arch=compute_35 cuda_fft.cu -lcublas

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

typedef float2 Complex;

#define DIM 65536

int main(int argc, char **argv) {
    float elapsedTime = 0;
    cufftHandle plan;
    cufftComplex *host_data = (cufftComplex*)malloc(DIM * sizeof(cufftComplex));
    cufftComplex *dev_data;
    cudaEvent_t start, stop;

    // feed input
    srand(time(NULL));
    for (int i = 0; i < DIM; i++) {
        host_data[i].x = rand() / (float) RAND_MAX;
        host_data[i].y = 0.0;
    }

    // allocate gpu memory
    cudaMalloc((void**)&dev_data, sizeof(cufftComplex) * DIM);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy input to device
    cudaMemcpy(dev_data, host_data, DIM * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    printf("test done!\n");
    return 0;
}