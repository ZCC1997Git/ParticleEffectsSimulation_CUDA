#ifndef __config_h__
#define __config_h__

/**
 * The number of values a pixel can take
 * This has no reason to be changed
 * Any change would likely necessitate changes to the code
 */
#define PIXEL_RANGE 256
/**
 * Number of runs to complete for benchmarking
 */
#define BENCHMARK_RUNS 100

/**
 * Particle generation Config
 */
#define CIRCLE_OPACITY_AVERAGE 0.5f
#define CIRCLE_OPACITY_STDDEV 2.0f 
#define CIRCLE_RAD_AVERAGE 10.0f 
#define CIRCLE_RAD_STDDEV 2.0f 

/**
 * Bounds for particle generation
 */
#define MIN_RADIUS 10.0f
#define MAX_RADIUS 512.0f
#define MIN_OPACITY 0.2f
#define MAX_OPACITY 0.8f

// Dark2 palette from Colorbrewer
static const unsigned char base_color_palette[8][3] = {
    {29, 143, 100},
    {206, 74, 8},
    {97, 89, 164},
    {222, 0, 119},
    {86, 153, 24},
    {223, 156, 9},
    {148, 99, 23},
    {83, 83, 83}
};

// Dependent config, do not change values hereafter
// f values are to save implicit/explicit casts in the code
// Some uses may ensure floating point division, be careful if replacing them


#endif  // __config_h__
