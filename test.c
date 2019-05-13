#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "self_openmp_fft_lib.h"
#include <fftw3.h>

#define N 8

int main() {

    // feed input
    fftw_complex *fftw_input;
    Complex *self_input;
    fftw_input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)* N);
    self_input = (Complex*) malloc(sizeof(Complex)* N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        fftw_input[i][0] = (float)i;
        fftw_input[i][1] = 0.0;
        self_input[i].a = (float)i;
        self_input[i].b = 0.0;
    }

    // perform fftw 1d fft
    fftw_complex * fftw_output;
    fftw_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)* N);
    fftw_plan my_plan;
    my_plan = fftw_plan_dft_1d(N, fftw_input, fftw_output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(my_plan);

    // perform self 1d fft
    Complex *self_output;
    self_output = (Complex*) malloc(sizeof(Complex)* N);
    openmp_1d_fft(self_input, self_output, N, FFT_FORWARD);

    // compare result
    printf("=======fftw result=======\n");
    for (int i = 0; i < 4; i++) {
        printf("DATA: %3.1f %3.1f\n", fftw_output[i][0], fftw_output[i][1]);
    }
    printf("======self result=======\n");
    for (int i = 0; i < 4; i++) {
        printf("DATA: %3.1f %3.1f\n", self_output[i].a, self_output[i].b);
    }

    return 0;
}