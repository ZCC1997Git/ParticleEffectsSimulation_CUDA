cmake_minimum_required(VERSION 3.5)
project(ParticlesEffect LANGUAGES C  CUDA)
enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES off)

SET(CMAKE_BUILD_TYPE "Release")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ./src/*.cu ./src/*.c)
file(GLOB_RECURSE header CONFIGURE_DEPENDS ./src/*.h,./src/*.cuh)

find_package(OpenMP REQUIRED)

add_executable(Particles ${sources})
target_include_directories(Particles PUBLIC ./src/ ./)
target_link_libraries(Particles PUBLIC OpenMP::OpenMP_C)
enable_testing()

add_test(
  NAME CPU_test
  COMMAND $<TARGET_FILE:Particles> CPU 10000 1024*1024 cpu.png
)
add_test(
  NAME OpenMP_test
  COMMAND $<TARGET_FILE:Particles> OPENMP 10000 1024*1024 omp.png
)

add_test(
  NAME CUDA_test
  COMMAND $<TARGET_FILE:Particles> CUDA 10000 1024*1024 cuda.png
)

