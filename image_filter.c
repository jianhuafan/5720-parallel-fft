#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;
    unit8_int* rgb_image = stbi_load("image/dog.jpg", &width, &height, &bpp, 3);

    printf("width: %d\n", width);
    printf("height: %d\n", height);
    printf("bpp: %d\n", bpp);

    stbi_image_free(rgb_image);

    return 0;
}