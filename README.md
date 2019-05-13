compile command

compile cuda_fft.cu
```
$ /usr/local/cuda-10.0/bin/nvcc -arch=compute_35 cuda_fft.cu -lcublas -lcufft
```

compile self_openmpfft.c
```
$ cc self_openmp_fft.c -fopenmp -lm
```

compile fftw_serial.c
```
$ gcc fftw_serial.c -fopenmp -lfftw3 -lm
```