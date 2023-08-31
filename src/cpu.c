#include "cpu.h"
#include "helper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

///
/// Utility Methods
///
/**
 * Simple (in place) quicksort pair sort implementation
 * Items at the matching index within keys and values are sorted according to
 * keys in ascending order
 * @param keys_start Pointer to the start of the keys (depth) buffer
 * @param colours_start Pointer to the start of the values (colours) buffer
 * (each color consists of three unsigned char)
 * @param first Index of the first item to be sorted
 * @param last Index of the last item to be sorted
 * @note This function is implemented at the bottom of cpu.c
 */
void cpu_sort_pairs(float* keys_start,
                    unsigned char* colours_start,
                    int first,
                    int last);

///
/// Algorithm storage
///
unsigned int cpu_particles_count;
Particle* cpu_particles;
unsigned int* cpu_pixel_contribs;
unsigned int* cpu_pixel_index;
unsigned char* cpu_pixel_contrib_colours;
float* cpu_pixel_contrib_depth;
unsigned int cpu_pixel_contrib_count;
CImage cpu_output_image;

///
/// Implementation
///
void cpu_begin(const Particle* init_particles,
               const unsigned int init_particles_count,
               const unsigned int out_image_width,
               const unsigned int out_image_height) {
    // Allocate a opy of the initial particles, to be used during computation
    cpu_particles_count = init_particles_count;
    cpu_particles = malloc(init_particles_count * sizeof(Particle));
    memcpy(cpu_particles, init_particles,
           init_particles_count * sizeof(Particle));

    // Allocate a histogram to track how many particles contribute to each pixel
    cpu_pixel_contribs = (unsigned int*)malloc(
        out_image_width * out_image_height * sizeof(unsigned int));
    // Allocate an index to track where data for each pixel's contributing
    // colour starts/ends
    cpu_pixel_index = (unsigned int*)malloc(
        (out_image_width * out_image_height + 1) * sizeof(unsigned int));
    // Init a buffer to store colours contributing to each pixel into (allocated
    // in stage 2)
    cpu_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into
    // (allocated in stage 2)
    cpu_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated
    // for, init 0
    cpu_pixel_contrib_count = 0;

    // Allocate output image
    cpu_output_image.width = (int)out_image_width;
    cpu_output_image.height = (int)out_image_height;
    cpu_output_image.channels = 3;  // RGB
    cpu_output_image.data = (unsigned char*)malloc(
        cpu_output_image.width * cpu_output_image.height *
        cpu_output_image.channels * sizeof(unsigned char));
}
void cpu_stage1() {
    // Reset the pixel contributions histogram
    memset(cpu_pixel_contribs, 0,
           cpu_output_image.width * cpu_output_image.height *
               sizeof(unsigned int));
    // Update each particle & calculate how many particles contribute to each
    // image
    for (unsigned int i = 0; i < cpu_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min =
            (int)roundf(cpu_particles[i].location[0] - cpu_particles[i].radius);
        int y_min =
            (int)roundf(cpu_particles[i].location[1] - cpu_particles[i].radius);
        int x_max =
            (int)roundf(cpu_particles[i].location[0] + cpu_particles[i].radius);
        int y_max =
            (int)roundf(cpu_particles[i].location[1] + cpu_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= cpu_output_image.width ? cpu_output_image.width - 1
                                                : x_max;
        y_max = y_max >= cpu_output_image.height ? cpu_output_image.height - 1
                                                 : y_max;
        // For each pixel in the bounding box, check that it falls within the
        // radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab =
                    (float)x + 0.5f - cpu_particles[i].location[0];
                const float y_ab =
                    (float)y + 0.5f - cpu_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= cpu_particles[i].radius) {
                    const unsigned int pixel_offset =
                        y * cpu_output_image.width + x;
                    ++cpu_pixel_contribs[pixel_offset];
                }
            }
        }
    }
#ifdef VALIDATION
    validate_pixel_contribs(cpu_particles, cpu_particles_count,
                            cpu_pixel_contribs, cpu_output_image.width,
                            cpu_output_image.height);
#endif
}

void cpu_stage2() {
    // Exclusive prefix sum across the histogram to create an index
    cpu_pixel_index[0] = 0;
    for (int i = 0; i < cpu_output_image.width * cpu_output_image.height; ++i) {
        cpu_pixel_index[i + 1] = cpu_pixel_index[i] + cpu_pixel_contribs[i];
    }
    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS =
        cpu_pixel_index[cpu_output_image.width * cpu_output_image.height];
    if (TOTAL_CONTRIBS > cpu_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (cpu_pixel_contrib_colours) free(cpu_pixel_contrib_colours);
        if (cpu_pixel_contrib_depth) free(cpu_pixel_contrib_depth);
        cpu_pixel_contrib_colours =
            (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        cpu_pixel_contrib_depth =
            (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        cpu_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    // Reset the pixel contributions histogram
    memset(cpu_pixel_contribs, 0,
           cpu_output_image.width * cpu_output_image.height *
               sizeof(unsigned int));
    // Store colours according to index
    // For each particle, store a copy of the colour/depth in cpu_pixel_contribs
    // for each contributed pixel
    for (unsigned int i = 0; i < cpu_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min =
            (int)roundf(cpu_particles[i].location[0] - cpu_particles[i].radius);
        int y_min =
            (int)roundf(cpu_particles[i].location[1] - cpu_particles[i].radius);
        int x_max =
            (int)roundf(cpu_particles[i].location[0] + cpu_particles[i].radius);
        int y_max =
            (int)roundf(cpu_particles[i].location[1] + cpu_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= cpu_output_image.width ? cpu_output_image.width - 1
                                                : x_max;
        y_max = y_max >= cpu_output_image.height ? cpu_output_image.height - 1
                                                 : y_max;
        // Store data for every pixel within the bounding box that falls within
        // the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab =
                    (float)x + 0.5f - cpu_particles[i].location[0];
                const float y_ab =
                    (float)y + 0.5f - cpu_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= cpu_particles[i].radius) {
                    const unsigned int pixel_offset =
                        y * cpu_output_image.width + x;
                    // Offset into cpu_pixel_contrib buffers is index +
                    // histogram Increment cpu_pixel_contribs, so next
                    // contributor stores to correct offset
                    const unsigned int storage_offset =
                        cpu_pixel_index[pixel_offset] +
                        (cpu_pixel_contribs[pixel_offset]++);
                    // Copy data to cpu_pixel_contrib buffers
                    memcpy(cpu_pixel_contrib_colours + (4 * storage_offset),
                           cpu_particles[i].color, 4 * sizeof(unsigned char));
                    memcpy(cpu_pixel_contrib_depth + storage_offset,
                           &cpu_particles[i].location[2], sizeof(float));
                }
            }
        }
    }

    // Pair sort the colours contributing to each pixel based on ascending depth
    for (int i = 0; i < cpu_output_image.width * cpu_output_image.height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        cpu_sort_pairs(cpu_pixel_contrib_depth, cpu_pixel_contrib_colours,
                       cpu_pixel_index[i], cpu_pixel_index[i + 1] - 1);
    }
#ifdef VALIDATION
    validate_pixel_index(cpu_pixel_contribs, cpu_pixel_index,
                         cpu_output_image.width, cpu_output_image.height);
    validate_sorted_pairs(cpu_particles, cpu_particles_count, cpu_pixel_index,
                          cpu_output_image.width, cpu_output_image.height,
                          cpu_pixel_contrib_colours, cpu_pixel_contrib_depth);
#endif
}
void cpu_stage3() {
    // Memset output image data to 255 (white)
    memset(cpu_output_image.data, 255,
           cpu_output_image.width * cpu_output_image.height *
               cpu_output_image.channels * sizeof(unsigned char));

    // Order dependent blending into output image
    for (int i = 0; i < cpu_output_image.width * cpu_output_image.height; ++i) {
        for (unsigned int j = cpu_pixel_index[i]; j < cpu_pixel_index[i + 1];
             ++j) {
            // Blend each of the red/green/blue colours according to the below
            // blend formula dest = src * opacity + dest * (1 - opacity);
            const float opacity =
                (float)cpu_pixel_contrib_colours[j * 4 + 3] / (float)255;
            cpu_output_image.data[(i * 3) + 0] =
                (unsigned char)((float)cpu_pixel_contrib_colours[j * 4 + 0] *
                                    opacity +
                                (float)cpu_output_image.data[(i * 3) + 0] *
                                    (1 - opacity));
            cpu_output_image.data[(i * 3) + 1] =
                (unsigned char)((float)cpu_pixel_contrib_colours[j * 4 + 1] *
                                    opacity +
                                (float)cpu_output_image.data[(i * 3) + 1] *
                                    (1 - opacity));
            cpu_output_image.data[(i * 3) + 2] =
                (unsigned char)((float)cpu_pixel_contrib_colours[j * 4 + 2] *
                                    opacity +
                                (float)cpu_output_image.data[(i * 3) + 2] *
                                    (1 - opacity));
            // cpu_pixel_contrib_colours is RGBA
            // cpu_output_image.data is RGB (final output image does not have an
            // alpha channel!)
        }
    }
#ifdef VALIDATION
    validate_blend(cpu_pixel_index, cpu_pixel_contrib_colours,
                   &cpu_output_image);
#endif
}
void cpu_end(CImage* output_image) {
    // Store return value
    output_image->width = cpu_output_image.width;
    output_image->height = cpu_output_image.height;
    output_image->channels = cpu_output_image.channels;
    memcpy(output_image->data, cpu_output_image.data,
           cpu_output_image.width * cpu_output_image.height *
               cpu_output_image.channels * sizeof(unsigned char));
    // Release allocations
    free(cpu_pixel_contrib_depth);
    free(cpu_pixel_contrib_colours);
    free(cpu_output_image.data);
    free(cpu_pixel_index);
    free(cpu_pixel_contribs);
    free(cpu_particles);
    // Return ptrs to nullptr
    cpu_pixel_contrib_depth = 0;
    cpu_pixel_contrib_colours = 0;
    cpu_output_image.data = 0;
    cpu_pixel_index = 0;
    cpu_pixel_contribs = 0;
    cpu_particles = 0;
}

void cpu_sort_pairs(float* keys_start,
                    unsigned char* colours_start,
                    const int first,
                    const int last) {
    // Based on
    // https://www.tutorialspoint.com/explain-the-quick-sort-technique-in-c-language
    int i, j, pivot;
    float depth_t;
    unsigned char color_t[4];
    if (first < last) {
        pivot = first;
        i = first;
        j = last;
        while (i < j) {
            while (keys_start[i] <= keys_start[pivot] && i < last)
                i++;
            while (keys_start[j] > keys_start[pivot])
                j--;
            if (i < j) {
                // Swap key
                depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;
                // Swap color
                memcpy(color_t, colours_start + (4 * i),
                       4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * i), colours_start + (4 * j),
                       4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * j), color_t,
                       4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = depth_t;
        // Swap color
        memcpy(color_t, colours_start + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * pivot), colours_start + (4 * j),
               4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        cpu_sort_pairs(keys_start, colours_start, first, j - 1);
        cpu_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}
