#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load("image/dog.jpg", &width, &height, &bpp, 3);

    printf("width: %d\n", width);
    printf("height: %d\n", height);
    printf("bpp: %d\n", bpp);

    printf("%i\n", sizeof(rgb_image[0]));

    // int i, j;
    // for (i = 0; i < height; i++) {
    //     for (j = 0; j < width; j++) {
    //         printf("%hhu\n", rgb_image[i * width + j]);
    //     }
    //     printf("\n");
    // }
    stbi_image_free(rgb_image);

    return 0;
}