#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "/usr/local/cuda-10.0/lib64/cufft.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load("image/dog.jpg", &width, &height, &bpp, 3);

    printf("width: %d\n", width);
    printf("height: %d\n", height);
    printf("bpp: %d\n", bpp);

    int i, j;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            uint8_t* pixel = rgb_image + (i * width + j) * bpp;
            printf("r: %hhu ", pixel[0]);
            printf("g: %hhu ", pixel[1]);
            printf("b: %hhu\n", pixel[2]);
        }
        printf("\n");
    }

    stbi_image_free(rgb_image);

    return 0;
}