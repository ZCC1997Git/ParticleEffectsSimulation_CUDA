# Particle Effects simulation

This is  based on code for COM4521/COM6521 assignment.

Modern video games regularly use particle effects to viually represent phenomena such as smoke and fire. Hundreds or thounsands of translucent particles, represented by either coloured shaped or sprite billboards, are renderen each frame with thire overlap blended to create the desired effect.

## Code implementation

### OpenMP

The implementation of the OpenMP version is similar to that of the CPU as a whole, and the specific implementation is as follows：

1. openmp_begin()
running time of openmp_begin() is small so that we keep it serial.

2. openmp_stage1()
We use the directive #pragma omp parallel for to parallelize the outermost loop, and all variables are set to be shared . As for the shared variable openmp_pixel_contribs, the directive #pragma omp atomic is used to ensure the correctness.

3. openmp_stage2()
The running time of openmp_stage2() can account for more than  half of total running time of CPU version.
There are dependencies in the first loop, and the computation is tiny compared with the other two loops, we don’t parallelize it.
To parallelize the second and the last loop, we use #prgama omp parallel to fork threads, and #pragma omp do to distribute iterations of loops to threads. Atomic operation is used in the second loop to ensure the correctness of final result, and #pragma barrier is used to Synchronize threads before the computation of the last loop.

4. openmp_stage3()
a simple #pragma omp parallel for is adopted directly because there are no dependencies in the loop.

5 openmp_end()
The function openmp_end() is kept the same as that of CPU version

### CUDA

1. cuda_begin
In cuda_begin, lobal memory on GPU are allocarted to store data. And D_OUT_IMAGE_WIDTH and D_OUT_IMAGE_HEIGHT are allocated in the constant memory on GPU. The corresponding data are copied into GPU memory after allocation,.

2. Cuda_stage1
To achieve cuda_stage1, we define GPU kernels: Memset, which sets variables in GPU memory to specific values, and Stage1, which completes the actual calculation of cuda_stage1().
We add a function named Memsetvalue. In this function, all computation is almost divided evenly among threads.
In the Stage1 kernel, we divide the computational load between threads based on the number of particles and use CUDA atomic operations to maintain correctness. In addition, pixel_contribs is copied back to the CPU for computational pixel_index.

3. Cuda_stage2
In cuda_stage2, we first calculate the pixel_index on the CPU and copy the result to the GPU. We also define two CUDA kernels, Stage2_0() and Stage2_1(), to complete the main calculations of the cuda_stage2.
The implementation of Stage2_0() is almost identical to the implementation of Stage1(), and we will not repeat it.
Stage2_1() is used to sort pixle_contrib_colours using a quicksort algorithm. We distribute the entire computation among threads based on the number of pixels. Each iteration calls quicksort, which we rewrite as a device function.

4. Cuda_stage3
We also define a GPU core for cuda_stage3(), which is relatively simple to implement. The entire loop is evenly broken down to all threads.

## using method

* make or make debug
* ./bin/release/Particles <mode> <particle count> <output image dimensions> (<output image>) (--bench)
Required Arguments:
<mode>             The algorithm to use: CPU, OPENMP, CUDA
<particle count>   The number of particles to generate
<output image dimensions> The dimensions of the image to output e.g. 512 or 512x1024
Optional Arguments:
<output image>     Output image, requires .png filetype
-b, --bench        Enable benchmark mode
* example
./bin/release/Particles CUDA 100 512*512 1.png

## result

./bin/release/Particles CUDA 100 512*512 1.png
![Alt text](1.png)
