#ifndef OPENMP_H_
#define OPENMP_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The initialisation function for the OpenMP Particles implementation
 * Memory allocation and initialisation occurs here, so that it can be timed separate to the algorithm
 * @param init_particles Pointer to an array of particle structures
 * @param init_particles_count The number of elements within the particles array
 * @param out_image_width The width of the final image to be output
 * @param out_image_height The height of the final image to be output
 */
void openmp_begin(const Particle* init_particles, unsigned int init_particles_count,
    unsigned int out_image_width, unsigned int out_image_height);
/**
 * Create a locatlised histogram for each tile of the image
 */
void openmp_stage1();
/**
 * Equalise the histograms
 */
void openmp_stage2();
/**
 * Interpolate the histograms to construct the contrast enhanced image for output
 */
void openmp_stage3();
/**
 * The cleanup and return function for the CPU CLAHE implemention
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be output, output_image->data is pre-allocated
 */
void openmp_end(CImage *output_image);

#ifdef __cplusplus
}
#endif

#endif  // OPENMP_H_
