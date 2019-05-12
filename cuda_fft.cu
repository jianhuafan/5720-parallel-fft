// compile command: /usr/local/cuda-10.0/bin/nvcc -arch=compute_35 cuda_fft.cu -lcublas -lcufft

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void ComplexMul(cufftComplex *a, cufftComplex *b, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads) {
        cufftComplex c;
        c.x = a[i].x * b[i].x - a[i].y*b[i].y;
        c.y = a[i].x * b[i].y + a[i].y*b[i].x;
        c.x /= (1.0f / size);
        c.y /= (1.0f / size);
        a[i] = c;
    }
}

int main(int argc, char **argv) {

    // load image
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load("image/dog.jpg", &width, &height, &bpp, STBI_grey);

    float elapsedTime = 0;
    cufftHandle plan;
    int mem_size = width * height * sizeof(cufftComplex);

    cufftComplex *signal = (cufftComplex*)malloc(mem_size);
    cufftComplex *filter_kernel = (cufftComplex*)malloc(mem_size);
    cufftComplex *dev_signal;
    cufftComplex *dev_filter_kernel;
    cudaEvent_t start, stop;

    // feed input
    srand(time(NULL));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint8_t* pixel = rgb_image + (i * width + j);
            signal[i * width + j].x = (float)pixel[0];
            signal[i * width + j].y = 0.0;
        }
    }

    // feed kernel
    for (int i = 0; i < height * width; i++) {
        filter_kernel[i].x = rand() / (float) RAND_MAX;
        filter_kernel[i].y = 0.0;
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
    cufftPlan2d(&plan, height, width, CUFFT_C2C);

    // perform 2dfft
    cufftExecC2C(plan, dev_signal, dev_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, dev_filter_kernel, dev_filter_kernel, CUFFT_FORWARD);

    // perform multiplication
    ComplexMul <<<32, 256>>>(dev_signal, dev_filter_kernel, width * height);

    // perform inverse 2dfft
    cufftExecC2C(plan, dev_signal, dev_signal, CUFFT_INVERSE);

    // copy back results
    cudaMemcpy(signal, dev_signal, mem_size, cudaMemcpyDeviceToHost);

    // get calculation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // show results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f \n", signal[i * 4 + j].x, signal[i * 4 + j].y);
        }
    }
    printf("CUFFT calculation completed: %5.3f ms\n", elapsedTime);

    // write filtered image
    uint8_t* output_rgb_image;
    output_rgb_image = (uint8_t*)malloc(width*height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output_rgb_image[i * width + j] = (uint8_t)signal[i * width + j].x;
            if (i < 4 && j < 4) {
                printf("%hhu\n", output_rgb_image[i * width + j]);
            }
        }
    }

    stbi_write_png("image/filtered_dog.png", width, height, 1, output_rgb_image, width);

    // free memory
    cufftDestroy(plan);
    cudaFree(dev_signal);
    cudaFree(dev_filter_kernel);
    free(signal);
    free(filter_kernel);
    stbi_image_free(rgb_image);

    return 0;
}