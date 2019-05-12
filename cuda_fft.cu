// compile command: /usr/local/cuda-10.0/bin/nvcc -arch=compute_35 cuda_fft.cu -lcublas

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

typedef float2 Complex;

int main(int argc, char **argv) {
    printf("cufft lib is usable");
    return 0;
}