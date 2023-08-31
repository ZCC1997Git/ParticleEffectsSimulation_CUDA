#ifndef CUDA_CUH_
#define CUDA_CUH_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

/**
 * The initialisation function for the CUDA Particles implementation
 * Memory allocation and initialisation occurs here, so that it can be timed
 * separate to the algorithm
 * @param init_particles Pointer to an array of particle structures
 * @param init_particles_count The number of elements within the particles array
 * @param out_image_width The width of the final image to be output
 * @param out_image_height The height of the final image to be output
 */
void cuda_begin(const Particle* init_particles,
                unsigned int init_particles_count,
                unsigned int out_image_width,
                unsigned int out_image_height);
/**
 * Create a locatlised histogram for each tile of the image
 */
void cuda_stage1();
/**
 * Equalise the histograms
 */
void cuda_stage2();
/**
 * Interpolate the histograms to construct the contrast enhanced image for
 * output
 */
void cuda_stage3();
/**
 * The cleanup and return function for the CPU CLAHE implemention
 * Memory should be freed, and the final image copied to output_image
 * @param output_image Pointer to a struct to store the final image to be
 * output, output_image->data is pre-allocated
 */
void cuda_end(CImage* output_image);

/**
 * Error check function for safe CUDA API calling
 * Wrap all calls to CUDA API functions with CUDA_CALL() to catch errors on
 * failure e.g. CUDA_CALL(cudaFree(myPtr)); CUDA_CHECk() can also be used to
 * perform error checking after kernel launches and async methods e.g.
 * CUDA_CHECK()
 */
#if defined(_DEBUG) || defined(D_DEBUG)
#define CUDA_CALL(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() \
    { gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__); }
#else
#define CUDA_CALL(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() \
    { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        if (line >= 0) {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line,
                    cudaGetErrorString(code));
        } else {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line,
                    cudaGetErrorString(code));
        }
        exit(EXIT_FAILURE);
    }
}

#endif  // CUDA_CUH_
