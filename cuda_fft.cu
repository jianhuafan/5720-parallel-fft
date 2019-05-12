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

int main(int argc, char **argv) {
    float elapsedTime = 0;
    cufftHandle plan;
    cufftComplex *host_data = (cufftComplex*)malloc(N*N * sizeof(cufftComplex));
    cufftComplex *dev_data;
    cudaEvent_t start, stop;

    // feed input
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; k++) {
            host_data[i * N + j].x = rand() / (float) RAND_MAX;
            host_data[i * N + j].y = 0.0;
        }
    }

    // allocate gpu memory
    cudaMalloc((void**)&dev_data, sizeof(cufftComplex) * N*N);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy input to device
    cudaMemcpy(dev_data, host_data, N*N * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    // create cufft plan
    cufftPlan2D(&plan, N, N, CUFFT_C2C, 1);

    // perform computation
    cufftExecC2C(plan, dev_data, dev_data, CUFFT_FORWARD);

    // copy back results
    cudaMemcpy(host_data, dev_data, sizeof(cufftComplex) * N*N, cudaMemcpyDeviceToHost);

    // get calculation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // show results
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("DATA: %3.1f %3.1f \n", host_data[i].x, host_data[i].y);
        }
    }
    printf("CUFFT calculation completed: %5.3f ms\n", elapsedTime);

    // free memory
    cufftDestroy(plan);
    cudaFree(dev_data);
    free(host_data);

    return 0;
}