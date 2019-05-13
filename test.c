#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "self_openmp_fft_lib.h"
#include <fftw3.h>

int main() {

    // load image
    int width, height, bpp;
    uint8_t* grey_image = stbi_load("input/sheep.png", &width, &height, &bpp, STBI_grey);
    printf("input image, width: %d, height: %d\n", width, height);

    // feed input
    fftw_complex *fftw_input;
    Complex *self_input;
    fftw_input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)* height*width);
    self_input = (Complex*) malloc(sizeof(Complex)* height*width);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            uint8_t* pixel = grey_image + (i * width + j);
            self_input[i * width + j].a = (double)pixel[0];
            self_input[i * width + j].b = 0.0;
            fftw_input[i * width + j][0] = (double)pixel[0];
            fftw_input[i * width + j][1] = 0.0;
        }
    }

    // perform fftw 2d fft
    fftw_complex * fftw_output;
    fftw_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)* height*width);
    fftw_plan my_plan;
    my_plan = fftw_plan_dft_2d(height, width, fftw_input, fftw_output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(my_plan);

    // perform self 2d fft
    Complex *self_output;
    self_output = (Complex*) malloc(sizeof(Complex)* height*width);
    openmp_2d_fft(self_input, self_output, height, width, FFT_BACKWARD);

    // compare result
    printf("=======fftw result=======\n");
    for (int i = 0; i < 8; i++) {
        printf("DATA: %3.1f %3.1f\n", fftw_output[i][0], fftw_output[i][1]);
    }
    printf("======self result=======\n");
    for (int i = 0; i < 8; i++) {
        printf("DATA: %3.1f %3.1f\n", self_output[i].a, self_output[i].b);
    }

    int wrong_result = 0;
    for (int i = 0; i < 8; i++) {
        if (abs(fftw_output[i][0] - self_output[i].a) > 1e-5) {
            wrong_result = 1;
            break;
        }
        if (abs(fftw_output[i][1] - self_output[i].b) > 1e-5) {
            wrong_result = 1;
            break;
        }
    }
    if (wrong_result) {
        printf("wrong result!!!!!!!!\n");
    }
    return 0;
}