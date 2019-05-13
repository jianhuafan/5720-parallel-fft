compile command

compile cuda_image_filter.cu
```
$ /usr/local/cuda-10.0/bin/nvcc -arch=compute_35 cuda_image_filter.cu -lcublas -lcufft
```

compile self_image_filter.c
```
$ cc self_image_filter.c -fopenmp -lm
```

compile fftw_image_filter.c
```
$ gcc fftw_image_filter.c -fopenmp -lfftw3 -lm
```