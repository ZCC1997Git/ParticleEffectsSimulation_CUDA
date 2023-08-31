#include <cmath>
#include <cstring>
#include "cuda.cuh"
#include "helper.h"

unsigned int cuda_particles_count;
Particle* d_particles;
unsigned int* d_pixel_contribs;
unsigned int* d_pixel_index;
unsigned char* d_pixel_contrib_colours;
float* d_pixel_contrib_depth;
unsigned int cuda_pixel_contrib_count;
int cuda_output_image_width;
int cuda_output_image_height;
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
unsigned char* d_output_image_data;

unsigned int* host_pixel_contribs;
unsigned int* host_pixel_index;
unsigned char* host_pixel_contrib_colours;
float* host_pixel_contrib_depth;

__global__ void Stage1(int particles_count,
                       Particle* d_particles,
                       unsigned int* d_pixel_contribs);

__global__ void Stage2_0(int particles_count,
                         Particle* d_particles,
                         unsigned int* d_pixel_contribs,
                         unsigned int* d_pixel_index,
                         unsigned char* d_pixel_contrib_colours,
                         float* d_pixel_contrib_depth);

__global__ void Stage2_1(float* d_pixel_contrib_depth,
                         unsigned char* d_pixel_contrib_colours,
                         unsigned int* d_pixel_index);
__global__ void Stage3(unsigned int* d_pixel_index,
                       unsigned char* d_pixel_contrib_colours,
                       unsigned char* d_output_image_data);

template <class T>
__global__ void MemSet(T* buf, T val, unsigned int count);

__device__ __host__ void device_sort_pairs(float* keys_start,
                                           unsigned char* colours_start,
                                           const int first,
                                           const int last);

void cuda_begin(const Particle* init_particles,
                const unsigned int init_particles_count,
                const unsigned int out_image_width,
                const unsigned int out_image_height) {
    cuda_particles_count = init_particles_count;
    CUDA_CALL(
        cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles,
                         init_particles_count * sizeof(Particle),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height *
                                                sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc(
        &d_pixel_index,
        (out_image_width * out_image_height + 1) * sizeof(unsigned int)));

    d_pixel_contrib_colours = 0;
    d_pixel_contrib_depth = 0;
    cuda_pixel_contrib_count = 0;

    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width,
                                 sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT,
                                 &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&d_output_image_data,
                         cuda_output_image_width * cuda_output_image_height *
                             CHANNELS * sizeof(unsigned char)));

    // Allocate a histogram to track how many particles contribute to each pixel
    host_pixel_contribs = (unsigned int*)malloc(
        out_image_width * out_image_height * sizeof(unsigned int));
    // Allocate an index to track where data for each pixel's contributing
    // colour starts/ends
    host_pixel_index = (unsigned int*)malloc(
        (out_image_width * out_image_height + 1) * sizeof(unsigned int));
}

void cuda_stage1() {
    MemSet<<<16, 512>>>(d_pixel_contribs, (unsigned int)0,
                        cuda_output_image_width * cuda_output_image_height);
    Stage1<<<16, 512>>>(cuda_particles_count, d_particles, d_pixel_contribs);
    CUDA_CALL(cudaMemcpy(host_pixel_contribs, d_pixel_contribs,
                         cuda_output_image_width * cuda_output_image_height *
                             sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
}

void cuda_stage2() {
    host_pixel_index[0] = 0;
    for (int i = 0; i < cuda_output_image_width * cuda_output_image_height;
         ++i) {
        host_pixel_index[i + 1] = host_pixel_index[i] + host_pixel_contribs[i];
    }

    const unsigned int TOTAL_CONTRIBS =
        host_pixel_index[cuda_output_image_width * cuda_output_image_height];

    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        if (d_pixel_contrib_colours)
            CUDA_CALL(cudaFree(d_pixel_contrib_colours));
        if (d_pixel_contrib_depth) CUDA_CALL(cudaFree(d_pixel_contrib_depth));
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours,
                             TOTAL_CONTRIBS * 4 * sizeof(unsigned char)));
        CUDA_CALL(
            cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    // Reset the pixel contributions histogram
    CUDA_CALL(
        cudaMemcpy(d_pixel_index, host_pixel_index,
                   (cuda_output_image_width * cuda_output_image_height + 1) *
                       sizeof(unsigned int),
                   cudaMemcpyHostToDevice));
    MemSet<<<16, 512>>>(d_pixel_contribs, (unsigned int)0,
                        cuda_output_image_width * cuda_output_image_height);
    Stage2_0<<<16, 512>>>(cuda_particles_count, d_particles, d_pixel_contribs,
                          d_pixel_index, d_pixel_contrib_colours,
                          d_pixel_contrib_depth);

    if (cuda_particles_count <= 20000)
        Stage2_1<<<16, 512>>>(d_pixel_contrib_depth, d_pixel_contrib_colours,
                              d_pixel_index);
    else {
        /*avoid stack overflow*/
        host_pixel_contrib_depth =
            (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        host_pixel_contrib_colours =
            (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        CUDA_CALL(cudaMemcpy(host_pixel_contrib_depth, d_pixel_contrib_depth,
                             TOTAL_CONTRIBS * sizeof(float),
                             cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(host_pixel_contrib_colours,
                             d_pixel_contrib_colours,
                             TOTAL_CONTRIBS * 4 * sizeof(unsigned char),
                             cudaMemcpyDeviceToHost));

#pragma omp parallel for
        for (int i = 0; i < cuda_output_image_width * cuda_output_image_height;
             ++i) {
            // Pair sort the colours which contribute to a single pigment
            device_sort_pairs(host_pixel_contrib_depth,
                              host_pixel_contrib_colours, host_pixel_index[i],
                              host_pixel_index[i + 1] - 1);
        }

        CUDA_CALL(cudaMemcpy(d_pixel_contrib_depth, host_pixel_contrib_depth,
                             TOTAL_CONTRIBS * sizeof(float),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_pixel_contrib_colours,
                             host_pixel_contrib_colours,
                             TOTAL_CONTRIBS * 4 * sizeof(unsigned char),
                             cudaMemcpyHostToDevice));
        free(host_pixel_contrib_depth);
        free(host_pixel_contrib_colours);
    }
}

void cuda_stage3() {
    const int CHANNELS = 3;
    MemSet<<<16, 512>>>(
        d_output_image_data, (unsigned char)255,
        cuda_output_image_width * cuda_output_image_height * CHANNELS);
    Stage3<<<16, 512>>>(d_pixel_index, d_pixel_contrib_colours,
                        d_output_image_data);
}

void cuda_end(CImage* output_image) {
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data,
                         cuda_output_image_width * cuda_output_image_height *
                             CHANNELS * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
}

template <class T>
__global__ void MemSet(T* buf, T val, unsigned int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_thread = gridDim.x * blockDim.x;
    int local_count = (count + num_thread - 1) / num_thread;

    for (unsigned int i = 0; i < local_count; ++i) {
        int index = tid + num_thread * i;
        if (index < count) buf[index] = val;
    }
}

/*Only one dimension*/
__global__ void Stage1(int particles_count,
                       Particle* d_particles,
                       unsigned int* d_pixel_contribs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_thread = gridDim.x * blockDim.x;
    int local_count = (particles_count + num_thread - 1) / num_thread;

    // Update each particle & calculate how many particles contribute to each
    // image
    for (unsigned int P = 0; P < local_count; ++P) {
        int i = tid + P * num_thread;
        if (i < particles_count) {
            // Compute bounding box [inclusive-inclusive]
            int x_min =
                (int)roundf(d_particles[i].location[0] - d_particles[i].radius);
            int y_min =
                (int)roundf(d_particles[i].location[1] - d_particles[i].radius);
            int x_max =
                (int)roundf(d_particles[i].location[0] + d_particles[i].radius);
            int y_max =
                (int)roundf(d_particles[i].location[1] + d_particles[i].radius);

            // Clamp bounding box to image bounds
            x_min = x_min < 0 ? 0 : x_min;
            y_min = y_min < 0 ? 0 : y_min;
            x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1
                                                  : x_max;
            y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1
                                                   : y_max;

            // For each pixel in the bounding box, check that it falls within
            // the radius
            for (int x = x_min; x <= x_max; ++x) {
                for (int y = y_min; y <= y_max; ++y) {
                    const float x_ab =
                        (float)x + 0.5f - d_particles[i].location[0];
                    const float y_ab =
                        (float)y + 0.5f - d_particles[i].location[1];
                    const float pixel_distance =
                        sqrtf(x_ab * x_ab + y_ab * y_ab);
                    if (pixel_distance <= d_particles[i].radius) {
                        const unsigned int pixel_offset =
                            y * D_OUTPUT_IMAGE_WIDTH + x;
                        // ++host_pixel_contribs[pixel_offset];
                        atomicAdd(&d_pixel_contribs[pixel_offset], 1);
                    }
                }
            }
        }
    }
}

__global__ void Stage2_0(int particles_count,
                         Particle* d_particles,
                         unsigned int* d_pixel_contribs,
                         unsigned int* d_pixel_index,
                         unsigned char* d_pixel_contrib_colours,
                         float* d_pixel_contrib_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_thread = gridDim.x * blockDim.x;
    int local_count = (particles_count + num_thread - 1) / num_thread;

    for (unsigned int P = 0; P < local_count; ++P) {
        int i = tid + P * num_thread;
        if (i < particles_count) {
            // Compute bounding box [inclusive-inclusive]
            int x_min =
                (int)roundf(d_particles[i].location[0] - d_particles[i].radius);
            int y_min =
                (int)roundf(d_particles[i].location[1] - d_particles[i].radius);
            int x_max =
                (int)roundf(d_particles[i].location[0] + d_particles[i].radius);
            int y_max =
                (int)roundf(d_particles[i].location[1] + d_particles[i].radius);

            // Clamp bounding box to image bounds
            x_min = x_min < 0 ? 0 : x_min;
            y_min = y_min < 0 ? 0 : y_min;
            x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1
                                                  : x_max;
            y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1
                                                   : y_max;

            for (int x = x_min; x <= x_max; ++x) {
                for (int y = y_min; y <= y_max; ++y) {
                    const float x_ab =
                        (float)x + 0.5f - d_particles[i].location[0];
                    const float y_ab =
                        (float)y + 0.5f - d_particles[i].location[1];
                    const float pixel_distance =
                        sqrtf(x_ab * x_ab + y_ab * y_ab);
                    if (pixel_distance <= d_particles[i].radius) {
                        const unsigned int pixel_offset =
                            y * D_OUTPUT_IMAGE_WIDTH + x;
                        unsigned int count =
                            atomicAdd(&d_pixel_contribs[pixel_offset], 1);
                        const unsigned int storage_offset =
                            d_pixel_index[pixel_offset] + count;
                        memcpy(d_pixel_contrib_colours + (4 * storage_offset),
                               d_particles[i].color, 4 * sizeof(unsigned char));
                        memcpy(d_pixel_contrib_depth + storage_offset,
                               &d_particles[i].location[2], sizeof(float));
                    }
                }
            }
        }
    }
}

__global__ void Stage2_1(float* d_pixel_contrib_depth,
                         unsigned char* d_pixel_contrib_colours,
                         unsigned int* d_pixel_index) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_thread = gridDim.x * blockDim.x;

    int allcount = D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT;

    int local_count = (allcount + num_thread - 1) / num_thread;

    for (int i = 0; i < local_count; ++i) {
        int index = tid + i * num_thread;
        if (index < allcount) {
            device_sort_pairs(d_pixel_contrib_depth, d_pixel_contrib_colours,
                              d_pixel_index[index],
                              d_pixel_index[index + 1] - 1);
        }
    }
}

__global__ void Stage3(unsigned int* d_pixel_index,
                       unsigned char* d_pixel_contrib_colours,
                       unsigned char* d_output_image_data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_thread = gridDim.x * blockDim.x;

    int allcount = D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT;

    int local_count = (allcount + num_thread - 1) / num_thread;

    for (int P = 0; P < local_count; ++P) {
        int i = tid + P * num_thread;
        if (i < allcount) {
            for (unsigned int j = d_pixel_index[i]; j < d_pixel_index[i + 1];
                 ++j) {
                const float opacity =
                    (float)d_pixel_contrib_colours[j * 4 + 3] / (float)255;
                d_output_image_data[(i * 3) + 0] =
                    (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 0] *
                                        opacity +
                                    (float)d_output_image_data[(i * 3) + 0] *
                                        (1 - opacity));
                d_output_image_data[(i * 3) + 1] =
                    (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 1] *
                                        opacity +
                                    (float)d_output_image_data[(i * 3) + 1] *
                                        (1 - opacity));
                d_output_image_data[(i * 3) + 2] =
                    (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 2] *
                                        opacity +
                                    (float)d_output_image_data[(i * 3) + 2] *
                                        (1 - opacity));
            }
        }
    }
}

__device__ __host__ void device_sort_pairs(float* keys_start,
                                           unsigned char* colours_start,
                                           const int first,
                                           const int last) {
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
        device_sort_pairs(keys_start, colours_start, first, j - 1);
        device_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}
