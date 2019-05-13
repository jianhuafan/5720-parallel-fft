#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "self_openmp_fft_lib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FILTER_KERNEL_SIZE 9
#define BILLION 1000000000L

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
            const Complex *filter_kernel, Complex **padded_filter_kernel,
            int filter_kernel_size) {
  int minRadius = filter_kernel_size / 2;
  int maxRadius = filter_kernel_size - minRadius;
  int new_size = signal_size + maxRadius;

  // Pad signal
  Complex *new_data =(Complex *)(malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
  memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
  *padded_signal = new_data;

  // Pad filter
  new_data = (Complex *)(malloc(sizeof(Complex) * new_size));
  memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
  memset(new_data + maxRadius, 0,
         (new_size - filter_kernel_size) * sizeof(Complex));
  memcpy(new_data + new_size - minRadius, filter_kernel,
         minRadius * sizeof(Complex));
  *padded_filter_kernel = new_data;

  return new_size;
}

void feed_gaussian_kernel(Complex *filter_kernel, int filter_kernel_size) {
    int i;
    for (i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i].a = 0.0;
        filter_kernel[i].b = 0.0;
    }
    filter_kernel[0].a = 1.0f / 16;
    filter_kernel[1].a = 2.0f / 16;
    filter_kernel[2].a = 1.0f / 16;
    filter_kernel[3].a = 2.0f / 16;
    filter_kernel[4].a = 4.0f / 16;
    filter_kernel[5].a = 2.0f / 16;
    filter_kernel[6].a = 1.0f / 16;
    filter_kernel[7].a = 2.0f / 16;
    filter_kernel[8].a = 1.0f / 16;
}

void feed_identity_kernel(Complex *filter_kernel, int filter_kernel_size) {
    int i;
    for (i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i].a = 0.0;
        filter_kernel[i].b = 0.0;
        if (i == FILTER_KERNEL_SIZE / 2) {
            filter_kernel[i].a = 1.0;
        }
    }
}

void feed_edge_detection_kernel(Complex *filter_kernel, int filter_kernel_size) {
    int i;
    for (i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i].a = 0.0;
        filter_kernel[i].b = 0.0;
    }
    filter_kernel[0].a = -1.0f;
    filter_kernel[1].a = -1.0f;
    filter_kernel[2].a = -1.0f;
    filter_kernel[3].a = -1.0f;
    filter_kernel[4].a = 8.0f;
    filter_kernel[5].a = -1.0f;
    filter_kernel[6].a = -1.0f;
    filter_kernel[7].a = -1.0f;
    filter_kernel[8].a = -1.0f;
}

void feed_box_blur_kernel(Complex *filter_kernel, int filter_kernel_size) {
    int i;
    for (i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i].a = 1.0 / filter_kernel_size;
        filter_kernel[i].b = 0.0;
    }
}

void feed_sharpen_kernel(Complex *filter_kernel, int filter_kernel_size) {
    int i;
    for (i = 0; i < FILTER_KERNEL_SIZE; i++) {
        filter_kernel[i].a = 0.0;
        filter_kernel[i].b = 0.0;
    }
    filter_kernel[1].a = -1.0f;
    filter_kernel[3].a = -1.0f;
    filter_kernel[4].a = 5.0f;
    filter_kernel[5].a = -1.0f;
    filter_kernel[7].a = -1.0f;
}

void ComplexMul(Complex *c1, Complex *c2, int size) {
    int i;
    for (i = 0; i < size; i++) {
        Complex c;
        c.a = c1[i].a * c2[i].a - c1[i].b*c2[i].b;
        c.b = c1[i].a * c2[i].b + c1[i].b*c2[i].a;
        c.a /= (double)size;
        c.b /= (double)size;
        c1[i].a = c.a;
        c1[i].b = c.b;
    }
}

int main(int argc, char **argv) {
    // load image
    int width, height, bpp;
    uint8_t* grey_image = stbi_load("input/sheep.png", &width, &height, &bpp, STBI_grey);
    printf("input image, width: %d, height: %d\n", width, height);

    Complex *signal;
    Complex *filter_kernel;
    signal = (Complex*) malloc(sizeof(Complex)* height * width);
    filter_kernel = (Complex*) malloc(sizeof(Complex)* FILTER_KERNEL_SIZE);

    // feed input
    srand(time(NULL));
    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            uint8_t* pixel = grey_image + (i * width + j);
            signal[i * width + j].a = (double)pixel[0];
            signal[i * width + j].b = 0.0;
        }
    }

    // feed kernel
    feed_sharpen_kernel(filter_kernel, FILTER_KERNEL_SIZE);

    // pad image and filter kernel
    Complex *padded_signal;
    Complex *padded_filter_kernel;
    int new_size = PadData(signal, &padded_signal, width * height, filter_kernel,
              &padded_filter_kernel, FILTER_KERNEL_SIZE);

    //have output buffer
    Complex *out_signal;
    Complex *out_filter_kernel;
    out_signal = (Complex*) malloc(sizeof(Complex)* new_size);
    out_filter_kernel = (Complex*) malloc(sizeof(Complex)* new_size);

    // print result
    printf("padded_signal\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", padded_signal[i * 4 + j].a, padded_signal[i * 4 + j].b);
        }
    }

    // print result
    printf("padded_filter_kernel\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", padded_filter_kernel[i * 4 + j].a, padded_filter_kernel[i * 4 + j].b);
        }
    }

    // convolution starts
    struct timespec start, end;
    long long unsigned int diff;
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */

    // perform 2d fft
    openmp_2d_fft(padded_signal, out_signal, height, width, FFT_FORWARD);
    openmp_2d_fft(padded_filter_kernel, out_filter_kernel, height, width, FFT_FORWARD);

    // print result
    printf("out_signal\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", out_signal[i * 4 + j].a, out_signal[i * 4 + j].b);
        }
    }

    // print result
    printf("out_filter_kernel\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", out_filter_kernel[i * 4 + j].a, out_filter_kernel[i * 4 + j].b);
        }
    }

    // perform multiplication
    ComplexMul(out_signal, out_filter_kernel, new_size);

    // print result
    printf("out_signal\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", out_signal[i * 4 + j].a, out_signal[i * 4 + j].b);
        }
    }

    // perform inverse fft
    openmp_2d_fft(out_signal, padded_signal, height, width, FFT_BACKWARD);

    // convolution ends
    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu us\n", (long long unsigned int) (diff / 1000));

    // print result
    printf("results\n");
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            printf("DATA: %3.1f %3.1f\n", padded_signal[i * 4 + j].a, padded_signal[i * 4 + j].b);
        }
    }

    // write filtered image
    uint8_t* output_grey_image;
    output_grey_image = (uint8_t*)malloc(width*height);
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            output_grey_image[i * width + j] = (uint8_t)padded_signal[i * width + j].a;
            if (i < 4 && j < 4) {
                printf("%hhu\n", output_grey_image[i * width + j]);
            }
        }
    }
    int result = stbi_write_png("output/self/filtered_sharpen_sheep.png", width, height, 1, output_grey_image, width);
    if (!result) {
        printf("error writing image!\n");
    }

    // free memory
    free(signal);
    free(padded_signal);
    free(filter_kernel);
    free(padded_filter_kernel);
    free(out_signal);
    free(out_filter_kernel);
    stbi_image_free(grey_image);
    stbi_image_free(output_grey_image);

    return 0;
}