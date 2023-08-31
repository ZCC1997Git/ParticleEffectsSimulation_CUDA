;
#include <ctype.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <random>
#include <set>

// Presumes all coloured IO in this file is to stdout
#define CONSOLE_RED isatty(fileno(stdout)) ? "\x1b[91m" : ""
#define CONSOLE_GREEN isatty(fileno(stdout)) ? "\x1b[92m" : ""
#define CONSOLE_YELLOW isatty(fileno(stdout)) ? "\x1b[93m" : ""
#define CONSOLE_RESET isatty(fileno(stdout)) ? "\x1b[39m" : ""

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

#include "common.h"
#include "config.h"
#include "cpu.h"
#include "cuda.cuh"
#include "helper.h"
#include "main.h"
#include "openmp.h"

int main(int argc, char** argv) {
#ifdef _MSC_VER
    {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD consoleMode;
        GetConsoleMode(hConsole, &consoleMode);
        consoleMode |=
            ENABLE_VIRTUAL_TERMINAL_PROCESSING;  // Enable support for ANSI
                                                 // colours (Windows 10+)
        SetConsoleMode(hConsole, consoleMode);
    }
#endif
    // Parse args
    Config config;
    parse_args(argc, argv, &config);

    // Generate Initial Particles from user_config
    const unsigned int particles_count = config.circle_count;
    Particle* particles = (Particle*)malloc(particles_count * sizeof(Particle));
    {
        // Random engine with a fixed seed and several distributions to be used
        std::mt19937 rng(12);
        std::uniform_real_distribution<float> normalised_float_dist(0, 1);
        std::normal_distribution<float> circle_rad_dist(CIRCLE_RAD_AVERAGE,
                                                        CIRCLE_RAD_STDDEV);
        std::normal_distribution<float> circle_opacity_dist(
            CIRCLE_OPACITY_AVERAGE, CIRCLE_OPACITY_STDDEV);
        std::uniform_int_distribution<int> color_palette_dist(
            0, sizeof(base_color_palette) / sizeof(unsigned char[3]) - 1);
        std::vector<float> depths(config.circle_count);
        depths[0] = 0;
        for (unsigned int i = 1; i < config.circle_count; ++i) {
            depths[i] = nextafterf(depths[i - 1], FLT_MAX);
        }
        shuffle(depths.begin(), depths.end(), rng);
        // Common
        for (unsigned int i = 0; i < config.circle_count; ++i) {
            const int palette_index = color_palette_dist(rng);
            particles[i].color[0] = base_color_palette[palette_index][0];
            particles[i].color[1] = base_color_palette[palette_index][1];
            particles[i].color[2] = base_color_palette[palette_index][2];
            particles[i].location[0] =
                normalised_float_dist(rng) * config.out_image_width;
            particles[i].location[1] =
                normalised_float_dist(rng) * config.out_image_height;
            particles[i].location[2] = depths[i];
            // Circle specific
            particles[i].radius = circle_rad_dist(rng);
            float t_opacity = circle_opacity_dist(rng);
            t_opacity = t_opacity < MIN_OPACITY ? MIN_OPACITY : t_opacity;
            t_opacity = t_opacity > MAX_OPACITY ? MAX_OPACITY : t_opacity;
            particles[i].color[3] = (unsigned char)(255 * t_opacity);
        }
        // Clamp radius to bounds (use OpenMP in an attempt to trigger OpenMPs
        // hidden init cost)
#pragma omp parallel for
        for (int i = 0; i < (int)particles_count; ++i) {
            particles[i].radius = particles[i].radius < MIN_RADIUS
                                      ? MIN_RADIUS
                                      : particles[i].radius;
            particles[i].radius = particles[i].radius > MAX_RADIUS
                                      ? MAX_RADIUS
                                      : particles[i].radius;
        }
    }

    // Create result for validation
    CImage validation_image;
    {
        // Init metadata
        validation_image.width = config.out_image_width;
        validation_image.height = config.out_image_height;
        validation_image.channels = 3;
        // Allocate memory
        validation_image.data = (unsigned char*)malloc(
            validation_image.width * validation_image.height *
            validation_image.channels * sizeof(unsigned char));
        // Allocate algorithm storage
        unsigned int* pixel_contribs = (unsigned int*)malloc(
            validation_image.width * validation_image.height *
            sizeof(unsigned int));
        ;
        unsigned int* pixel_index = (unsigned int*)malloc(
            (validation_image.width * validation_image.height + 1) *
            sizeof(unsigned int));
        // Run algorithm
        skip_pixel_contribs(particles, particles_count, pixel_contribs,
                            validation_image.width, validation_image.height);
        skip_pixel_index(pixel_contribs, pixel_index, validation_image.width,
                         validation_image.height);
        const unsigned int TOTAL_CONTRIBS =
            pixel_index[validation_image.width * validation_image.height];
        unsigned char* pixel_contrib_colours =
            (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        float* pixel_contrib_depth =
            (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        skip_sorted_pairs(particles, particles_count, pixel_index,
                          validation_image.width, validation_image.height,
                          pixel_contrib_colours, pixel_contrib_depth);
        skip_blend(pixel_index, pixel_contrib_colours, &validation_image);
        // Free algorithm storage
        free(pixel_contrib_depth);
        free(pixel_contrib_colours);
        free(pixel_index);
        free(pixel_contribs);
    }

    CImage output_image;
    Runtimes timing_log;
    const int TOTAL_RUNS = 100;  // config.benchmark ? BENCHMARK_RUNS : 1;
    {
        // Init for run
        cudaEvent_t startT, initT, stage1T, stage2T, stage3T, stopT;
        CUDA_CALL(cudaEventCreate(&startT));
        CUDA_CALL(cudaEventCreate(&initT));
        CUDA_CALL(cudaEventCreate(&stage1T));
        CUDA_CALL(cudaEventCreate(&stage2T));
        CUDA_CALL(cudaEventCreate(&stage3T));
        CUDA_CALL(cudaEventCreate(&stopT));

        // Run 1 or many times
        memset(&timing_log, 0, sizeof(Runtimes));
        for (int runs = 0; runs < TOTAL_RUNS; ++runs) {
            if (TOTAL_RUNS > 1) printf("\r%d/%d", runs + 1, TOTAL_RUNS);
            memset(&output_image, 0, sizeof(CImage));
            output_image.data = (unsigned char*)malloc(
                config.out_image_width * config.out_image_height * 3 *
                sizeof(unsigned char));
            memset(output_image.data, 0,
                   config.out_image_width * config.out_image_height * 3 *
                       sizeof(unsigned char));
            // Run Particles algorithm
            CUDA_CALL(cudaEventRecord(startT));
            CUDA_CALL(cudaEventSynchronize(startT));
            switch (config.mode) {
                case CPU: {
                    cpu_begin(particles, particles_count,
                              config.out_image_width, config.out_image_height);
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    cpu_stage1();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    cpu_stage2();
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    cpu_stage3();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    cpu_end(&output_image);
                } break;
                case OPENMP: {
                    openmp_begin(particles, particles_count,
                                 config.out_image_width,
                                 config.out_image_height);
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    openmp_stage1();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    openmp_stage2();
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    openmp_stage3();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    openmp_end(&output_image);
                } break;
                case CUDA: {
                    cuda_begin(particles, particles_count,
                               config.out_image_width, config.out_image_height);
                    CUDA_CHECK();
                    CUDA_CALL(cudaEventRecord(initT));
                    CUDA_CALL(cudaEventSynchronize(initT));
                    cuda_stage1();
                    CUDA_CHECK();
                    CUDA_CALL(cudaEventRecord(stage1T));
                    CUDA_CALL(cudaEventSynchronize(stage1T));
                    cuda_stage2();
                    CUDA_CHECK();
                    CUDA_CALL(cudaEventRecord(stage2T));
                    CUDA_CALL(cudaEventSynchronize(stage2T));
                    cuda_stage3();
                    CUDA_CHECK();
                    CUDA_CALL(cudaEventRecord(stage3T));
                    CUDA_CALL(cudaEventSynchronize(stage3T));
                    cuda_end(&output_image);
                } break;
            }
            CUDA_CALL(cudaEventRecord(stopT));
            CUDA_CALL(cudaEventSynchronize(stopT));
            // Sum timing info
            float milliseconds = 0;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, initT));
            timing_log.init += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, initT, stage1T));
            timing_log.stage1 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage1T, stage2T));
            timing_log.stage2 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage2T, stage3T));
            timing_log.stage3 += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, stage3T, stopT));
            timing_log.cleanup += milliseconds;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, stopT));
            timing_log.total += milliseconds;
            // Avoid memory leak
            if (runs + 1 < TOTAL_RUNS) {
                if (output_image.data) free(output_image.data);
            }
        }
        // Convert timing info to average
        timing_log.init /= TOTAL_RUNS;
        timing_log.stage1 /= TOTAL_RUNS;
        timing_log.stage2 /= TOTAL_RUNS;
        timing_log.stage3 /= TOTAL_RUNS;
        timing_log.cleanup /= TOTAL_RUNS;
        timing_log.total /= TOTAL_RUNS;

        // Cleanup timing
        cudaEventDestroy(startT);
        cudaEventDestroy(initT);
        cudaEventDestroy(stage1T);
        cudaEventDestroy(stage2T);
        cudaEventDestroy(stage3T);
        cudaEventDestroy(stopT);
    }

    // Validate and report
    {
        printf("\rValidation Status: \n");
        printf("\tImage width: %s%s%s\n",
               validation_image.width == output_image.width ? CONSOLE_GREEN
                                                            : CONSOLE_RED,
               validation_image.width == output_image.width ? "Pass" : "Fail",
               CONSOLE_RESET);
        printf("\tImage height: %s%s%s\n",
               validation_image.height == output_image.height ? CONSOLE_GREEN
                                                              : CONSOLE_RED,
               validation_image.height == output_image.height ? "Pass" : "Fail",
               CONSOLE_RESET);
        printf("\tImage channels: %s%s%s\n",
               validation_image.channels == output_image.channels
                   ? CONSOLE_GREEN
                   : CONSOLE_RED,
               validation_image.channels == output_image.channels ? "Pass"
                                                                  : "Fail",
               CONSOLE_RESET);
        const int v_size = validation_image.width * validation_image.height;
        const int o_size = output_image.width * output_image.height;
        const int s_size = v_size < o_size ? v_size : o_size;
        const int max_channels =
            validation_image.channels > output_image.channels
                ? output_image.channels
                : validation_image.channels;
        int bad_pixels = 0;
        int close_pixels = 0;
        if (output_image.data && s_size) {
            for (int i = 0; i < s_size; ++i) {
                for (int ch = 0; ch < max_channels; ++ch) {
                    if (output_image.data[i * max_channels + ch] !=
                        validation_image.data[i * max_channels + ch]) {
                        // Give a +-1 threshold for error (incase fast-math
                        // triggers a small difference in places)
                        if (output_image.data[i * max_channels + ch] + 1 ==
                                validation_image.data[i * max_channels + ch] ||
                            output_image.data[i * max_channels + ch] - 1 ==
                                validation_image.data[i * max_channels + ch]) {
                            close_pixels++;
                        } else {
                            bad_pixels++;
                        }
                        break;
                    }
                }
            }
            printf("\tImage pixels: ");
            if (bad_pixels) {
                printf("%sFail%s (%d/%u pixels contain the wrong colour)\n",
                       CONSOLE_RED, CONSOLE_RESET, bad_pixels, o_size);
            } else {
                printf("%sPass%s\n", CONSOLE_GREEN, CONSOLE_RESET);
            }
        } else {
            printf("\tImage pixels: %sFail%s\n", CONSOLE_RED, CONSOLE_RESET);
        }
    }

    // Export output image
    if (config.output_file) {
        if (!stbi_write_png(config.output_file, output_image.width,
                            output_image.height, output_image.channels,
                            output_image.data,
                            output_image.width * output_image.channels)) {
            printf("%sUnable to save image output to %s.%s\n", CONSOLE_YELLOW,
                   config.output_file, CONSOLE_RESET);
            // return EXIT_FAILURE;
        }
    }

    // Report timing information
    printf("%s Average execution timing from %d runs\n",
           mode_to_string(config.mode), TOTAL_RUNS);
    if (config.mode == CUDA) {
        int device_id = 0;
        CUDA_CALL(cudaGetDevice(&device_id));
        cudaDeviceProp props;
        memset(&props, 0, sizeof(cudaDeviceProp));
        CUDA_CALL(cudaGetDeviceProperties(&props, device_id));
        printf("Using GPU: %s\n", props.name);
    }
#ifdef _DEBUG
    printf("%sCode built as DEBUG, timing results are invalid!\n%s",
           CONSOLE_YELLOW, CONSOLE_RESET);
#endif
    printf("Init: %.3fms\n", timing_log.init);
    printf("Stage 1: %.3fms%s%s%s\n", timing_log.stage1,
           getStage1SkipUsed() ? CONSOLE_YELLOW : "",
           getStage1SkipUsed() ? " (helper method used, time invalid)" : "",
           CONSOLE_RESET);
    printf("Stage 2: %.3fms%s%s%s\n", timing_log.stage2,
           getStage2SkipUsed() ? CONSOLE_YELLOW : "",
           getStage2SkipUsed() ? " (helper method used, time invalid)" : "",
           CONSOLE_RESET);
    printf("Stage 3: %.3fms%s%s%s\n", timing_log.stage3,
           getStage3SkipUsed() ? CONSOLE_YELLOW : "",
           getStage3SkipUsed() ? " (helper method used, time invalid)" : "",
           CONSOLE_RESET);
    printf("Free: %.3fms\n", timing_log.cleanup);
    printf("Total: %.3fms%s%s%s\n", timing_log.total,
           getSkipUsed() ? CONSOLE_YELLOW : "",
           getSkipUsed() ? " (helper method used, time invalid)" : "",
           CONSOLE_RESET);

    // Cleanup
    cudaDeviceReset();
    free(validation_image.data);
    free(particles);
    free(output_image.data);
    if (config.output_file) free(config.output_file);
    return EXIT_SUCCESS;
}
void parse_args(int argc, char** argv, Config* config) {
    // Clear config struct
    memset(config, 0, sizeof(Config));
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "Program expects 3-5 arguments, only %d provided.\n",
                argc - 1);
        print_help(argv[0]);
    }
    // Parse first arg as mode
    {
        char lower_arg[7];  // We only care about first 6 characters
        // Convert to lower case
        int i = 0;
        for (; argv[1][i] && i < 6; i++) {
            lower_arg[i] = tolower(argv[1][i]);
        }
        lower_arg[i] = '\0';
        // Check for a match
        if (!strcmp(lower_arg, "cpu")) {
            config->mode = CPU;
        } else if (!strcmp(lower_arg, "openmp")) {
            config->mode = OPENMP;
        } else if (!strcmp(lower_arg, "cuda") || !strcmp(lower_arg, "gpu")) {
            config->mode = CUDA;
        } else {
            fprintf(stderr,
                    "Unexpected string provided as first argument: '%s' .\n",
                    argv[1]);
            fprintf(stderr,
                    "First argument expects a single mode as string: CPU, "
                    "OPENMP, CUDA.\n");
            print_help(argv[0]);
        }
    }
    // Parse second arg as number of particles
    {
        int particle_arg = atoi(argv[2]);
        if (particle_arg > 0) {
            config->circle_count = (unsigned int)particle_arg;
        } else {
            fprintf(stderr,
                    "Unexpected uint provided as second argument: '%s' .\n",
                    argv[2]);
            fprintf(stderr,
                    "Second argument expects the number of particles to "
                    "generate.\n");
            print_help(argv[0]);
        }
    }
    // Parse third arg as output image dimensions
    {
        // Attempt to parse as two uints, delimited by , or x or X
        const int in_width = atoi(strtok(argv[3], ",xX"));
        const char* in_height_str = strtok(NULL, ",xX");
        const int in_height = in_height_str ? atoi(in_height_str) : 0;
        const char* in_end = strtok(NULL, ",x");
        if (in_width > 0 && in_height > 0 &&
            in_end == NULL) {  // width + height provided
            config->out_image_width = (unsigned int)in_width;
            config->out_image_height = (unsigned int)in_height;
        } else if (in_width > 0 && in_height == 0 &&
                   in_end == NULL) {  // Only width provided
            config->out_image_width = (unsigned int)in_width;
            config->out_image_height = (unsigned int)in_width;
        } else {
            fprintf(
                stderr,
                "Unable to parse input provided as third argument: '%s' .\n",
                argv[2]);
            fprintf(stderr,
                    "Third argument expects the image dimensions (e.g. 512 or "
                    "512x1024).\n");
            print_help(argv[0]);
        }
    }

    // Iterate over remaining args
    int i = 4;
    char* t_arg = 0;
    for (; i < argc; i++) {
        // Make a lowercase copy of the argument
        const size_t arg_len =
            strlen(argv[i]) + 1;  // Add 1 for null terminating character
        if (t_arg) free(t_arg);
        t_arg = (char*)malloc(arg_len);
        int j = 0;
        for (; argv[i][j]; ++j) {
            t_arg[j] = tolower(argv[i][j]);
        }
        t_arg[j] = '\0';
        // Decide which arg it is
        if (!strcmp("--bench", t_arg) || !strcmp("--benchmark", t_arg) ||
            !strcmp("-b", t_arg)) {
            config->benchmark = 1;
            continue;
        }
        if (!strcmp(t_arg + arg_len - 5, ".png")) {
            // Allocate memory and copy
            config->output_file = (char*)malloc(arg_len);
            memcpy(config->output_file, argv[i], arg_len);
            continue;
        }
        fprintf(stderr, "Unexpected optional argument in position %d: %s\n", i,
                argv[i]);
        print_help(argv[0]);
    }
    if (t_arg) free(t_arg);
}
void print_help(const char* program_name) {
    fprintf(stderr,
            "%s <mode> <particle count> <output image dimensions> (<output "
            "image>) (--bench)\n",
            program_name);

    const char* line_fmt = "%-18s %s\n";
    fprintf(stderr, "Required Arguments:\n");
    fprintf(stderr, line_fmt, "<mode>",
            "The algorithm to use: CPU, OPENMP, CUDA");
    fprintf(stderr, line_fmt, "<particle count>",
            "The number of particles to generate");
    fprintf(stderr, line_fmt, "<output image dimensions>",
            "The dimensions of the image to output e.g. 512 or 512x1024");
    fprintf(stderr, "Optional Arguments:\n");
    fprintf(stderr, line_fmt, "<output image>",
            "Output image, requires .png filetype");
    fprintf(stderr, line_fmt, "-b, --bench", "Enable benchmark mode");

    exit(EXIT_FAILURE);
}
const char* mode_to_string(Mode m) {
    switch (m) {
        case CPU:
            return "CPU";
        case OPENMP:
            return "OpenMP";
        case CUDA:
            return "CUDA";
    }
    return "?";
}
