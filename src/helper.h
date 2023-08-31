#ifndef __helper_h__
#define __helper_h__

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These helper methods can be use to check validity of individual stages of your algorithm
 * Pass the required component to the validate_?() methods for it's validation status to be printed to console
 * Pass the required components to the skip_?() methods to have the reference implementation perform the step.
 * For the CUDA algorithm, this will require copying data back to the host. You CANNOT pass device pointers to these methods.
 *
 * Pointers passed to helper methods must point to memory which has been allocated in the same format as cpu.c
 *
 * Some inputs will have limited/no errors when performed incorrectly, it's best to validate with a wide range of inputs.
 *
 * Do not use these methods during benchmark runs, as they will invalidate the timing.
 */

///
/// Stage 1 helpers
///
/**
 * Validates whether the results of stage 1 have been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param particles The list of particles used to calculate test_pixel_contribs 
 * @param particles_count The length of the array behind pointer particles
 * @param test_pixel_contribs Pointer to the pixel contribution histogram
 * @param out_image_width The output image width (multiplies with out_image_height for the length of test_pixel_contribs)
 * @param out_image_height The output image height (multiplies with out_image_width for the length of test_pixel_contribs)
 *
 * @note If particles or test_pixel_contribs does not match the same memory layout as cpu.c, this may cause an access violation
 */
void validate_pixel_contribs(
    const Particle *particles, unsigned int particles_count,
    const unsigned int *test_pixel_contribs, unsigned int out_image_width, unsigned int out_image_height);
/**
 * Calculate the results of stage 1 from the input particles
 * The result is applied to the parameter return_pixel_contribs
 *
 * @param particles The list of particles used to calculate test_pixel_contribs
 * @param particles_count The length of the array behind pointer particles
 * @param return_pixel_contribs Pointer to allocated storage for the pixel contribution histogram
 * @param out_image_width The output image width (multiplies with out_image_height for the length of return_pixel_contribs)
 * @param out_image_height The output image height (multiplies with out_image_width for the length of return_pixel_contribs)
 */
void skip_pixel_contribs(
    const Particle *particles, unsigned int particles_count,
    unsigned int *return_pixel_contribs, unsigned int out_image_width, unsigned int out_image_height);

///
/// Stage 2 helpers
///
/**
 * Validates whether test_pixel_index has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param pixel_contribs Pointer to the pixel contribution histogram
 * @param test_pixel_index Pointer to the result of exclusive prefix sum over pixel_contribs
 * @param out_image_width The output image width (multiplies with out_image_height for the length of pixel_contribs)
 * @param out_image_height The output image height (multiplies with out_image_width for the length of pixel_contribs)
 *
 * @note The length of test_pixel_index should be out_image_width * out_image_height + 1
 */
void validate_pixel_index(const unsigned int *pixel_contribs, const unsigned int *test_pixel_index, unsigned int out_image_width, unsigned int out_image_height);
/**
 * Calculate pixel_index of stage 2 using pixel_contribs
 * The result is applied to the parameter return_pixel_index
 *
 * @param pixel_contribs Pointer to the pixel contribution histogram
 * @param return_pixel_index Pointer to the result of exclusive prefix sum over pixel_contribs
 * @param out_image_width The output image width (multiplies with out_image_height for the length of pixel_contribs)
 * @param out_image_height The output image height (multiplies with out_image_width for the length of pixel_contribs)
 *
 * @note The length of test_pixel_index should be out_image_width * out_image_height + 1
 */
void skip_pixel_index(const unsigned int *pixel_contribs, unsigned int *return_pixel_index, unsigned int out_image_width, unsigned int out_image_height);
/**
 * Validates whether test_pixel_contrib_colours and test_pixel_contrib_depth have been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param particles The list of particles used to calculate test_pixel_contribs 
 * @param particles_count The length of the array behind pointer particles
 * @param pixel_index Pointer to the result of exclusive prefix sum over pixel_contribs
 * @param out_image_width The output image width (multiplies with out_image_height for the length of pixel_contribs)
 * @param out_image_height The output image height (multiplies with out_image_width for the length of pixel_contribs)
 * @param test_pixel_contrib_colours The list of colours contributing to each pixel, colours for each pixel should be sorted according to depth
 * @param test_pixel_contrib_depth (Optional, 0 can be passed instead) The list of depths for colours contributing to each pixel, colours for each pixel should be sorted in ascending order
 *
 * @note If the allocated length of test_pixel_contrib_colours or test_pixel_contrib_depth is too short an access violation may occur
 */
void validate_sorted_pairs(
    const Particle *particles, unsigned int particles_count,
    const unsigned int *pixel_index, unsigned int out_image_width, unsigned int out_image_height,
    const unsigned char *test_pixel_contrib_colours, const float *test_pixel_contrib_depth);
/**
 * Calculate the sorted pairs of stage 2 using pixel_index and the particles
 * The result is applied to the parameter return_pixel_contrib_colours and return_pixel_contrib_depth
 *
 * @param particles The list of particles used to calculate test_pixel_contribs
 * @param particles_count The length of the array behind pointer particles
 * @param pixel_index Pointer to the result of exclusive prefix sum over pixel_contribs
 * @param out_image_width The output image width (multiplies with out_image_height for the length of pixel_contribs)
 * @param out_image_height The output image height (multiplies with out_image_width for the length of pixel_contribs)
 * @param return_pixel_contrib_colours The list of colours contributing to each pixel, colours for each pixel should be sorted according to depth
 * @param return_pixel_contrib_depth The list of depths for colours contributing to each pixel, colours for each pixel should be sorted in ascending order
 *
 * @note If the allocated length of pixel_contrib_colours or pixel_contrib_depth is too short an access violation may occur
 */
void skip_sorted_pairs(
    const Particle *particles, unsigned int particles_count,
    const unsigned int *pixel_index, unsigned int out_image_width, unsigned int out_image_height,
    unsigned char *return_pixel_contrib_colours, float *return_pixel_contrib_depth);

///
/// Stage 3 helpers
///
/**
 * Validates whether the output image of stage 3 has been calculated correctly
 * Success or failure will be printed to the console
 *
 * @param pixel_index Pointer to the result of exclusive prefix sum over pixel_contribs
 * @param pixel_contrib_colours The list of colours contributing to each pixel, colours for each pixel should be sorted according to depth
 * @param test_output_image The final output image to be tested
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 * @note The length of pixel_index should be out_image_width * out_image_height + 1
 */
void validate_blend(const unsigned int *pixel_index, const unsigned char *pixel_contrib_colours, const CImage *test_output_image);
/**
 * Calculate the output image of stage 3 using pixel_index and pixel_contrib_colours from stage 2
 * The result is applied to the return_output_image histograms
 *
 * @param pixel_index Pointer to the result of exclusive prefix sum over pixel_contribs
 * @param pixel_contrib_colours The list of colours contributing to each pixel, colours for each pixel should be sorted according to depth
 * @param return_output_image Host pointer to a pre-allocated image for output
 *
 * @note If any of the input parameters do not point to memory matching the layout of cpu.c, this may cause an access violation
 */
void skip_blend(const unsigned int *pixel_index, const unsigned char *pixel_contrib_colours, CImage *return_output_image);

///
/// These are used for reporting whether timing is invalid due to helper use
///
int getSkipUsed();
int getStage1SkipUsed();
int getStage2SkipUsed();
int getStage3SkipUsed();

#ifdef __cplusplus
}
#endif

#if defined(_DEBUG) || defined(DEBUG)
#define VALIDATION
#endif

#endif  // __helper_h__
