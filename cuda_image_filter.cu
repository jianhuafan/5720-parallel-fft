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
        c.x /= size;
        c.y /= size;
        a[i] = c;
    }
}

// Pad data
int PadData(const cufftComplex *signal, cufftComplex **padded_signal, int signal_size,
            const cufftComplex *filter_kernel, cufftComplex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  cufftComplex *new_data =(cufftComplex *)(malloc(sizeof(cufftComplex) * new_size));
  memcpy(new_data + 0, signal, signal_size * sizeof(cufftComplex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(cufftComplex));
  *padded_signal = new_data;

  // Pad filter
  new_data = (cufftComplex *)(malloc(sizeof(cufftComplex) * new_size));
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(cufftComplex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(cufftComplex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(cufftComplex));
  *padded_filter_kernel = new_data;

  return new_size;
}

void feed_gaussian_kernel(cufftComplex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < filter_kernel_size; i++) {
        filter_kernel[i].x = 0.0;
        filter_kernel[i].y = 0.0;
    }
    filter_kernel[0].x = 1.0f / 16;
    filter_kernel[1].x = 2.0f / 16;
    filter_kernel[2].x = 1.0f / 16;
    filter_kernel[3].x = 2.0f / 16;
    filter_kernel[4].x = 4.0f / 16;
    filter_kernel[5].x = 2.0f / 16;
    filter_kernel[6].x = 1.0f / 16;
    filter_kernel[7].x = 2.0f / 16;
    filter_kernel[8].x = 1.0f / 16;
}

void feed_identity_kernel(cufftComplex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < filter_kernel_size; i++) {
        filter_kernel[i].x = 0.0;
        filter_kernel[i].y = 0.0;
        if (i == filter_kernel_size / 2) {
            filter_kernel[i].x = 1.0;
        }
    }
}

void feed_edge_detection_kernel(cufftComplex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < filter_kernel_size; i++) {
        filter_kernel[i].x = 0.0;
        filter_kernel[i].y = 0.0;
    }
    filter_kernel[0].x = -1.0f;
    filter_kernel[1].x = -1.0f;
    filter_kernel[2].x = -1.0f;
    filter_kernel[3].x = -1.0f;
    filter_kernel[4].x = 8.0f;
    filter_kernel[5].x = -1.0f;
    filter_kernel[6].x = -1.0f;
    filter_kernel[7].x = -1.0f;
    filter_kernel[8].x = -1.0f;
}

void feed_box_blur_kernel(cufftComplex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < filter_kernel_size; i++) {
        filter_kernel[i].x = 1.0 / filter_kernel_size;
        filter_kernel[i].y = 0.0;
    }
}

void feed_sharpen_kernel(cufftComplex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < filter_kernel_size; i++) {
        filter_kernel[i].x = 0.0;
        filter_kernel[i].y = 0.0;
    }
    filter_kernel[1].x = -1.0f;
    filter_kernel[3].x = -1.0f;
    filter_kernel[4].x = 5.0f;
    filter_kernel[5].x = -1.0f;
    filter_kernel[7].x = -1.0f;
}

int main(int argc, char **argv) {

    // load image
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load("input/256.png", &width, &height, &bpp, STBI_grey);

    int filter_kernel_size = 15 * 15;

    float elapsedTime = 0;
    cufftHandle plan;
    int signal_size = width * height * sizeof(cufftComplex);

    cufftComplex *signal = (cufftComplex*)malloc(signal_size);
    cufftComplex *filter_kernel = (cufftComplex*)malloc(filter_kernel_size * sizeof(cufftComplex));
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
    feed_identity_kernel(filter_kernel, filter_kernel_size);

    // pad image and filter kernel
    cufftComplex *padded_signal;
    cufftComplex *padded_filter_kernel;
    int new_size = PadData(signal, &padded_signal, width * height, filter_kernel,
              &padded_filter_kernel, filter_kernel_size);
    
    int mem_size = sizeof(cufftComplex) * new_size;

    // allocate gpu memory
    cudaMalloc((void**)&dev_signal, mem_size);
    cudaMalloc((void**)&dev_filter_kernel, mem_size);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // copy padded input to device
    cudaMemcpy(dev_signal, padded_signal, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filter_kernel, padded_filter_kernel, mem_size, cudaMemcpyHostToDevice);
    
    // create cufft plan
    cufftPlan2d(&plan, height, width, CUFFT_C2C);

    // perform 2dfft
    cufftExecC2C(plan, dev_signal, dev_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, dev_filter_kernel, dev_filter_kernel, CUFFT_FORWARD);

    // perform multiplication
    ComplexMul <<<32, 256>>>(dev_signal, dev_filter_kernel, new_size);

    // perform inverse 2dfft
    cufftExecC2C(plan, dev_signal, dev_signal, CUFFT_INVERSE);

    // copy back results
    cufftComplex *convolved_signal = padded_signal;
    cudaMemcpy(convolved_signal, dev_signal, mem_size, cudaMemcpyDeviceToHost);

    // get calculation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // show results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f \n", convolved_signal[i * 4 + j].x, convolved_signal[i * 4 + j].y);
        }
    }
    printf("CUFFT calculation completed: %5.3f ms\n", elapsedTime);

    // write filtered image
    uint8_t* output_rgb_image;
    output_rgb_image = (uint8_t*)malloc(width*height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output_rgb_image[i * width + j] = (uint8_t)convolved_signal[i * width + j].x;
            if (i < 4 && j < 4) {
                printf("%hhu\n", output_rgb_image[i * width + j]);
            }
        }
    }

    stbi_write_png("output/cuda/filtered_identity_256.png", width, height, 1, output_rgb_image, width);

    // free memory
    cufftDestroy(plan);
    cudaFree(dev_signal);
    cudaFree(dev_filter_kernel);
    free(signal);
    free(filter_kernel);
    stbi_image_free(rgb_image);

    return 0;
}
