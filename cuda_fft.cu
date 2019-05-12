// compile command: /usr/local/cuda-10.0/bin/nvcc -arch=compute_35 cuda_fft.cu -lcublas -lcufft

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

#define N 4

__global__ void ComplexMul(cufftComplex *a, cufftComplex *b) {
    int i = threadIdx.x;
    a[i].x = a[i].x * b[i].x - a[i].y*b[i].y;
    a[i].y = a[i].x * b[i].y + a[i].y*b[i].x;
}

int main(int argc, char **argv) {
    float elapsedTime = 0;
    cufftHandle plan;
    int mem_size = N*N * sizeof(cufftComplex);
    cufftComplex *signal = (cufftComplex*)malloc(mem_size);
    cufftComplex *filter_kernel = (cufftComplex*)malloc(mem_size);
    cufftComplex *dev_signal;
    cufftComplex *dev_filter_kernel;
    cudaEvent_t start, stop;

    // feed input
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            signal[i * N + j].x = rand() / (float) RAND_MAX;
            signal[i * N + j].y = 0.0;
            filter_kernel[i * N + j].x = rand() / (float) RAND_MAX;
            filter_kernel[i * N + j].y = 0.0;
        }
    }

    // allocate gpu memory
    cudaMalloc((void**)&dev_signal, mem_size);
    cudaMalloc((void**)&dev_filter_kernel, mem_size);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy input to device
    cudaMemcpy(dev_signal, signal, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filter_kernel, filter_kernel, mem_size, cudaMemcpyHostToDevice);
    
    // create cufft plan
    cufftPlan2d(&plan, N, N, CUFFT_C2C);

    // perform 2dfft
    cufftExecC2C(plan, dev_signal, dev_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, dev_filter_kernel, dev_filter_kernel, CUFFT_FORWARD);

    // perform multiplication
    ComplexMul <<<32, 256>>>(dev_signal, dev_filter_kernel);

    // perform inverse 2dfft
    cufftExecC2C(plan, dev_signal, dev_signal, CUFFT_INVERSE);

    // get calculation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // show results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("DATA: %3.1f %3.1f \n", signal[i].x, signal[i].y);
        }
    }
    printf("CUFFT calculation completed: %5.3f ms\n", elapsedTime);

    // free memory
    cufftDestroy(plan);
    cudaFree(dev_signal);
    cudaFree(dev_filter_kernel);
    free(signal);
    free(filter_kernel);

    return 0;
}