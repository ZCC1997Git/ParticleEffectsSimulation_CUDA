#include "helper.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#ifdef _MSC_VER
#pragma warning(disable: 4996)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "config.h"

// Presumes all coloured IO in this file is to stderr
#define CONSOLE_RED isatty(fileno(stderr))?"\x1b[91m":""
#define CONSOLE_GREEN isatty(fileno(stderr))?"\x1b[92m":""
#define CONSOLE_YELLOW isatty(fileno(stderr))?"\x1b[93m":""
#define CONSOLE_RESET isatty(fileno(stderr))?"\x1b[39m":""

///
/// Utility Methods
///
/**
 * Simple (in place) quicksort pair sort implementation
 * Items at the matching index within keys and values are sorted according to keys in ascending order
 * @param keys_start Pointer to the start of the keys (depth) buffer
 * @param colours_start Pointer to the start of the values (colours) buffer (each color consists of three unsigned char)
 * @param first Index of the first item to be sorted
 * @param last Index of the last item to be sorted
 * @note This function is implemented at the bottom of cpu.c
 */
void help_sort_pairs(float* keys_start, unsigned char* colours_start, int first, int last);


int skip_pixel_contribs_used = -1;
void validate_pixel_contribs(
    const Particle *particles, const unsigned int particles_count,
    const unsigned int *test_pixel_contribs, const unsigned int out_image_width, const unsigned int out_image_height) {
    // Allocate, copy and generate our own internal pixel_contribs
    unsigned int* pixel_contribs = (unsigned int*)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    skip_pixel_contribs(particles, particles_count, pixel_contribs, out_image_width, out_image_height);
    skip_pixel_contribs_used--;
    // Validate and report result
    unsigned int bad_contribs = 0;
    for (unsigned int i = 0; i < out_image_width * out_image_height; ++i) {
        if (test_pixel_contribs[i] != pixel_contribs[i]) {
            ++bad_contribs;
        }
    }
    if (bad_contribs) {
        fprintf(stderr, "validate_pixel_contribs() %sfound %d/%u pixels contain invalid values.%s\n", CONSOLE_RED, bad_contribs, out_image_width * out_image_height, CONSOLE_RESET);
    } else {
        fprintf(stderr, "validate_pixel_contribs() %sfound no errors! (%u pixels were correct)%s\n", CONSOLE_GREEN, out_image_width * out_image_height, CONSOLE_RESET);
    }
    // Release internal pixel_contribs
    free(pixel_contribs);
}
void skip_pixel_contribs(
    const Particle *particles, const unsigned int particles_count,
    unsigned int *return_pixel_contribs, const unsigned int out_image_width, const unsigned int out_image_height) {
    // Reset the pixel contributions histogram
    memset(return_pixel_contribs, 0, out_image_width * out_image_height * sizeof(unsigned int));
    // Update each particle & calculate how many particles contribute to each image
    for (unsigned int i = 0; i < particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(particles[i].location[0] - particles[i].radius);
        int y_min = (int)roundf(particles[i].location[1] - particles[i].radius);
        int x_max = (int)roundf(particles[i].location[0] + particles[i].radius);
        int y_max = (int)roundf(particles[i].location[1] + particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= (int)out_image_width ? (int)out_image_width - 1 : x_max;
        y_max = y_max >= (int)out_image_height ? (int)out_image_height - 1 : y_max;
        // For each pixel in the bounding box, check that it falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particles[i].location[0];
                const float y_ab = (float)y + 0.5f - particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particles[i].radius) {
                    const unsigned int pixel_offset = y * out_image_width + x;
                    ++return_pixel_contribs[pixel_offset];
                }
            }
        }
    }
    skip_pixel_contribs_used++;
}

int skip_pixel_index_used = -1;
void validate_pixel_index(const unsigned int *pixel_contribs, const unsigned int *test_pixel_index, const unsigned int out_image_width, const unsigned int out_image_height) {
    // Allocate, copy and generate our own internal pixel_index
    unsigned int* pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
    skip_pixel_index(pixel_contribs, pixel_index, out_image_width, out_image_height);
    skip_pixel_index_used--;
    // Validate and report result
    unsigned int bad_indices = 0;
    for (unsigned int i = 0; i < (out_image_width * out_image_height + 1); ++i) {
        if (test_pixel_index[i] != pixel_index[i]) {
            ++bad_indices;
        }
    }
    if (bad_indices) {
        fprintf(stderr, "validate_pixel_index() %sfound %d/%u pixels contain invalid indices.%s\n", CONSOLE_RED, bad_indices, out_image_width * out_image_height, CONSOLE_RESET);
    } else {
        fprintf(stderr, "validate_pixel_index() %sfound no errors! (%u pixels were correct)%s\n", CONSOLE_GREEN, out_image_width * out_image_height, CONSOLE_RESET);
    }
    // Release internal pixel_index
    free(pixel_index);    
}
void skip_pixel_index(const unsigned int *pixel_contribs, unsigned int *return_pixel_index, const unsigned int out_image_width, const unsigned int out_image_height) {
    // Exclusive prefix sum across the histogram to create an index
    return_pixel_index[0] = 0;
    for (unsigned int i = 0; i < out_image_width * out_image_height; ++i) {
        return_pixel_index[i + 1] = return_pixel_index[i] + pixel_contribs[i];
    }
    skip_pixel_index_used++;
}

int skip_sorted_pairs_used = -1;
void validate_sorted_pairs(
    const Particle *particles, const unsigned int particles_count,
    const unsigned int *pixel_index, const unsigned int out_image_width, const unsigned int out_image_height,
    const unsigned char *test_pixel_contrib_colours, const float *test_pixel_contrib_depth) {
    // Allocate, copy and generate our own internal pixel_index
    const unsigned int TOTAL_CONTRIBS = pixel_index[out_image_width * out_image_height];
    unsigned char* pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    float* pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
    skip_sorted_pairs(
        particles, particles_count,
        pixel_index, out_image_width, out_image_height,
        pixel_contrib_colours, pixel_contrib_depth);
    skip_sorted_pairs_used--;
    // Validate and report result
    unsigned int bad_colours = 0;
    unsigned int bad_depth = 0;
    for (unsigned int i = 0; i < out_image_width * out_image_height; ++i) {
        bool pixel_has_bad_color = false;
        bool pixel_has_bad_depth = false;
        for (unsigned int j = pixel_index[i]; j < pixel_index[i + 1]; ++j) {
            if (!pixel_has_bad_color && (
                test_pixel_contrib_colours[j * 4 + 0] != pixel_contrib_colours[j * 4 + 0] ||
                test_pixel_contrib_colours[j * 4 + 1] != pixel_contrib_colours[j * 4 + 1] ||
                test_pixel_contrib_colours[j * 4 + 2] != pixel_contrib_colours[j * 4 + 2] ||
                test_pixel_contrib_colours[j * 4 + 3] != pixel_contrib_colours[j * 4 + 3] )                
            ) {
                ++bad_colours;
                pixel_has_bad_color = true;
            }
            if (!pixel_has_bad_depth && test_pixel_contrib_depth &&
                test_pixel_contrib_depth[j] != pixel_contrib_depth[j]) { // Soften with epsilon?
                ++bad_depth;
                pixel_has_bad_depth = true;
            }
        }
    }
    if (bad_colours) {
        fprintf(stderr, "validate_sorted_pairs() %sfound %d/%u pixels have wrong/unsorted colours.%s\n", CONSOLE_RED, bad_colours, out_image_width * out_image_height, CONSOLE_RESET);
    } else {
        fprintf(stderr, "validate_sorted_pairs() %sfound no colour errors! (%u pixels colours were correct)%s\n", CONSOLE_GREEN, out_image_width * out_image_height, CONSOLE_RESET);
    }
    if (test_pixel_contrib_depth) {
        if (bad_depth) {
            fprintf(stderr, "validate_sorted_pairs() %sfound %d/%u pixels have wrong/unsorted depths.%s\n", CONSOLE_RED, bad_depth, out_image_width * out_image_height, CONSOLE_RESET);
            if (!bad_colours) {
                fprintf(stderr, "%sColours were correct, so incorrect depth is not a problem.%s\n", CONSOLE_YELLOW, CONSOLE_RESET);
            }
        } else {
            fprintf(stderr, "validate_sorted_pairs() %sfound no depth errors! (%u pixels depths were correct)%s\n", CONSOLE_GREEN, out_image_width * out_image_height, CONSOLE_RESET);
        }
    }
    // Release internal pixel contrib buffers
    free(pixel_contrib_colours);
    free(pixel_contrib_depth);
}
void skip_sorted_pairs(
    const Particle *particles, const unsigned int particles_count,
    const unsigned int *pixel_index, const unsigned int out_image_width, const unsigned int out_image_height,
    unsigned char *return_pixel_contrib_colours, float *return_pixel_contrib_depth) {
    // Allocate a temporary internal pixel contributions histogram
    unsigned int *pixel_contribs = (unsigned int *)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    memset(pixel_contribs, 0, out_image_width * out_image_height * sizeof(unsigned int));
    // Store colours according to index
    // For each particle, store a copy of the colour/depth in cpu_pixel_contribs for each contributed pixel
    for (unsigned int i = 0; i < particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(particles[i].location[0] - particles[i].radius);
        int y_min = (int)roundf(particles[i].location[1] - particles[i].radius);
        int x_max = (int)roundf(particles[i].location[0] + particles[i].radius);
        int y_max = (int)roundf(particles[i].location[1] + particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= (int)out_image_width ? (int)out_image_width - 1 : x_max;
        y_max = y_max >= (int)out_image_height ? (int)out_image_height - 1 : y_max;
        // Store data for every pixel within the bounding box that falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particles[i].location[0];
                const float y_ab = (float)y + 0.5f - particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particles[i].radius) {
                    const unsigned int pixel_offset = y * out_image_width + x;
                    // Offset into cpu_pixel_contrib buffers is index + histogram
                    // Increment cpu_pixel_contribs, so next contributor stores to correct offset
                    const unsigned int storage_offset = pixel_index[pixel_offset] + (pixel_contribs[pixel_offset]++);
                    // Copy data to cpu_pixel_contrib buffers
                    memcpy(return_pixel_contrib_colours + (4 * storage_offset), particles[i].color, 4 * sizeof(unsigned char));
                    memcpy(return_pixel_contrib_depth + storage_offset, &particles[i].location[2], sizeof(float));
                }
            }
        }
    }

    // Pair sort the colours contributing to each pixel based on ascending depth
    for (unsigned int i = 0; i < out_image_width * out_image_height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        help_sort_pairs(
            return_pixel_contrib_depth,
            return_pixel_contrib_colours,
            pixel_index[i],
            pixel_index[i + 1] - 1
        );
    }
    // Release temporary internal pixel contributions histogram
    free(pixel_contribs);
    skip_sorted_pairs_used++;
}


int skip_blend_used = -1;
void validate_blend(const unsigned int *pixel_index, const unsigned char *pixel_contrib_colours, const CImage *test_output_image) {
    // Allocate, copy and generate our own internal output image
    CImage output_image;
    memcpy(&output_image, test_output_image, sizeof(CImage));
    output_image.data = (unsigned char*)malloc(output_image.width * output_image.height * output_image.channels * sizeof(unsigned char));
    skip_blend(pixel_index, pixel_contrib_colours, &output_image);
    skip_blend_used--;
    // Validate and report result
    unsigned int bad_pixels = 0;
    unsigned int close_pixels = 0;
    for (int i = 0; i < output_image.width * output_image.height; ++i) {
        for (int ch = 0; ch < output_image.channels; ++ch) {
            if (output_image.data[i * output_image.channels + ch] != test_output_image->data[i * output_image.channels + ch]) {
                // Give a +-1 threshold for error (incase fast-math triggers a small difference in places)
                if (output_image.data[i * output_image.channels + ch]+1 == test_output_image->data[i * output_image.channels + ch] ||
                    output_image.data[i * output_image.channels + ch]-1 == test_output_image->data[i * output_image.channels + ch]) {
                    close_pixels++;
                } else {
                    bad_pixels++;
                }
                break;
            }
        }
    }
    if (output_image.channels != 3) {
        fprintf(stderr, "validate_blend() %sOutput image channels should equal 3, found %d instead.%s\n", CONSOLE_RED, output_image.channels, CONSOLE_RESET);
    }
    if (bad_pixels) {
        fprintf(stderr, "validate_blend() %s%d/%u pixels contain the wrong colour.%s\n", CONSOLE_RED, bad_pixels, output_image.width * output_image.height, CONSOLE_RESET);
    } else if(output_image.channels == 3){
        fprintf(stderr, "validate_blend() %sfound no errors! (%u pixels were correct)%s\n", CONSOLE_GREEN, output_image.width * output_image.height, CONSOLE_RESET);
    }
    // Release internal output image
    free(output_image.data);
}
void skip_blend(const unsigned int *pixel_index, const unsigned char *pixel_contrib_colours, CImage *return_output_image) {
    // Memset output image data to 255 (white)
    memset(return_output_image->data, 255, return_output_image->width * return_output_image->height * return_output_image->channels * sizeof(unsigned char));

    // Order dependent blending into output image
    for (int i = 0; i < return_output_image->width * return_output_image->height; ++i) {
        for (unsigned int j = pixel_index[i]; j < pixel_index[i + 1]; ++j) {
            // Blend each of the red/green/blue colours according to the below blend formula
            // dest = src * opacity + dest * (1 - opacity);
            const float opacity = (float)pixel_contrib_colours[j * 4 + 3] / (float)255;
            return_output_image->data[(i * 3) + 0] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 0] * opacity + (float)return_output_image->data[(i * 3) + 0] * (1 - opacity));
            return_output_image->data[(i * 3) + 1] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 1] * opacity + (float)return_output_image->data[(i * 3) + 1] * (1 - opacity));
            return_output_image->data[(i * 3) + 2] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 2] * opacity + (float)return_output_image->data[(i * 3) + 2] * (1 - opacity));
            // cpu_pixel_contrib_colours is RGBA
            // cpu_output_image.data is RGB (final output image does not have an alpha channel!)
        }
    }
    skip_blend_used++;
}

int getSkipUsed() {
    return skip_pixel_contribs_used + skip_pixel_index_used + skip_sorted_pairs_used + skip_blend_used;
}
int getStage1SkipUsed() {
    return skip_pixel_contribs_used;
}
int getStage2SkipUsed() {
    return skip_pixel_index_used + skip_sorted_pairs_used;
}
int getStage3SkipUsed() {
    return skip_blend_used;
}

void help_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
    // Based on https://www.tutorialspoint.com/explain-the-quick-sort-technique-in-c-language
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
                memcpy(color_t, colours_start + (4 * i), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * i), colours_start + (4 * j), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = depth_t;
        // Swap color
        memcpy(color_t, colours_start + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * pivot), colours_start + (4 * j), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        help_sort_pairs(keys_start, colours_start, first, j - 1);
        help_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}