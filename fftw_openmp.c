#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include <fftw3.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_KERNEL_SIZE 9
#define BILLION 1000000000L

// Pad data
int PadData(const fftw_complex *signal, fftw_complex **padded_signal, int signal_size,
            const fftw_complex *filter_kernel, fftw_complex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  fftw_complex *new_data =(fftw_complex *)(malloc(sizeof(fftw_complex) * new_size));
  memcpy(new_data + 0, signal, signal_size * sizeof(fftw_complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(fftw_complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = (fftw_complex *)(malloc(sizeof(fftw_complex) * new_size));
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(fftw_complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(fftw_complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(fftw_complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

void feed_gaussian_kernel(fftw_complex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i][0] = 0.0;
        filter_kernel[i][1] = 0.0;
    }
    filter_kernel[0][0] = 1.0f / 16;
    filter_kernel[1][0] = 2.0f / 16;
    filter_kernel[2][0] = 1.0f / 16;
    filter_kernel[3][0] = 2.0f / 16;
    filter_kernel[4][0] = 4.0f / 16;
    filter_kernel[5][0] = 2.0f / 16;
    filter_kernel[6][0] = 1.0f / 16;
    filter_kernel[7][0] = 2.0f / 16;
    filter_kernel[8][0] = 1.0f / 16;
}

void feed_identity_kernel(fftw_complex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i][0] = 0.0;
        filter_kernel[i][1] = 0.0;
        if (i == FILTER_KERNEL_SIZE / 2) {
            filter_kernel[i][0] = 1.0;
        }
    }
}

void feed_edge_detection_kernel(fftw_complex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i][0] = 0.0;
        filter_kernel[i][1] = 0.0;
    }
    filter_kernel[0][0] = -1.0f;
    filter_kernel[1][0] = -1.0f;
    filter_kernel[2][0] = -1.0f;
    filter_kernel[3][0] = -1.0f;
    filter_kernel[4][0] = 8.0f;
    filter_kernel[5][0] = -1.0f;
    filter_kernel[6][0] = -1.0f;
    filter_kernel[7][0] = -1.0f;
    filter_kernel[8][0] = -1.0f;
}

void feed_box_blur_kernel(fftw_complex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i][0] = 1.0 / filter_kernel_size;
        filter_kernel[i][1] = 0.0;
    }
}

void feed_sharpen_kernel(fftw_complex *filter_kernel, int filter_kernel_size) {
    for (int i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i][0] = 0.0;
        filter_kernel[i][1] = 0.0;
    }
    filter_kernel[1][0] = -1.0f;
    filter_kernel[3][0] = -1.0f;
    filter_kernel[4][0] = 5.0f;
    filter_kernel[5][0] = -1.0f;
    filter_kernel[7][0] = -1.0f;
}

void ComplexMul(fftw_complex *a, fftw_complex *b, int size) {
    for (int i = 0; i < size; i++) {
        fftw_complex c;
        c[0] = a[i][0] * b[i][0] - a[i][1]*b[i][1];
        c[1] = a[i][0] * b[i][1] + a[i][1]*b[i][0];
        c[0] /= size;
        c[1] /= size;
        a[i] = c;
    }
}

int main(int argc, char **argv) {
    // load image
    int width, height, bpp;
    uint8_t* grey_image = stbi_load("image/sheep.png", &width, &height, &bpp, STBI_grey);

    fftw_complex *signal;
    fftw_complex *filter_kernel;
    signal = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)* height * width);
    filter_kernel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)* FILTER_KERNEL_SIZE);

    // feed input
    srand(time(NULL));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint8_t* pixel = grey_image + (i * width + j);
            signal[i * width + j][0] = (double)pixel[0];
            signal[i * width + j][1] = 0.0;
        }
    }

    // feed kernel
    feed_identity_kernel(filter_kernel, FILTER_KERNEL_SIZE);

    // pad image and filter kernel
    fftw_complex *padded_signal;
    fftw_complex *padded_filter_kernel;
    int new_size = PadData(signal, &padded_signal, width * height, filter_kernel,
              &padded_filter_kernel, FILTER_KERNEL_SIZE);

    // convolution starts
    struct timespec start, end;
    long long unsigned int diff;
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

    // create plan
    fftw_plan signal_plan;
    fftw_plan kernel_plan;
    signal_plan = fftw_plan_dft_2d(height, width, padded_signal, padded_signal, FFTW_FORWARD, FFTW_ESTIMATE);
    kernel_plan = fftw_plan_dft_2d(height, width, padded_filter_kernel, padded_filter_kernel, FFTW_FORWARD, FFTW_ESTIMATE);

    // perform 2d fft
    fftw_execute(signal_plan);
    fftw_execute(kernel_plan);

    // perform multiplication
    ComplexMul(padded_signal, padded_filter_kernel, new_size);

    // perform inverse fft
    fftw_plan inverse_signal_plan;
    inverse_signal_plan = fftw_plan_dft_2d(height, width, padded_signal, padded_signal, FFTW_BACKWARD, FFTW_ESTIMATE);

    // convolution ends
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu us\n", (long long unsigned int) (diff / 1000));

    // print result
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", padded_signal[i * 4 + j][0], padded_signal[i * 4 + j][1]);
        }
    }

    // write filtered image
    uint8_t* output_grey_image;
    output_grey_image = (uint8_t*)malloc(width*height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output_grey_image[i * width + j] = (uint8_t)padded_signal[i * width + j][0];
            if (i < 4 && j < 4) {
                printf("%hhu\n", output_grey_image[i * width + j]);
            }
        }
    }
    stbi_write_png("image/fftw/filtered_sharpen_sheep.png", width, height, 1, output_grey_image, width);

    // free memory
    fftw_destroy_plan(signal_plan);
    fftw_destroy_plan(kernel_plan);
    fftw_free(signal);
    fftw_free(filter_kernel);
    fftw_free(padded_signal);
    fftw_free(padded_filter_kernel);

    return 0;
}