#ifndef COMMON_H_
#define COMMON_H_

#include "config.h"

/**
 * This structure represents a multi-channel image
 * It is used to hold the image data to be exported
 */
struct CImage {
    /**
     * Array of pixel data of the image, 1 unsigned char per pixel channel
     * Pixels ordered left to right, top to bottom
     * There is no stride, this is a compact storage
     */
    unsigned char* data;
    /**
     * Image width and height
     */
    int width, height;
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     */
    int channels;
};
typedef struct CImage CImage;

/**
 * Data required to render a single circle particle
 */
struct Particle {
    unsigned char color[4];
    float location[3];
    float radius;
};
typedef struct Particle Particle;


#endif  // COMMON_H_
