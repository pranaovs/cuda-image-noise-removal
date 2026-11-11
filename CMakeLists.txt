cmake_minimum_required(VERSION 3.16)
project(hpc_denoise C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -O2 is a host flag; nvcc takes -O3 from CMAKE_BUILD_TYPE=Release and warns about
# the duplicate, so scope it to the C compiler.
add_compile_options($<$<COMPILE_LANGUAGE:C>:-O2> -Wall -Wextra)

include_directories(${CMAKE_SOURCE_DIR}/src/common)

# CUDA lives behind a flag so a box without nvcc still builds everything CPU-side.
option(ENABLE_CUDA "Build the CUDA implementations (src/cuda)" OFF)
find_package(OpenMP)
find_library(M_LIB m)

add_library(common STATIC
    src/common/image.c
    src/common/noise.c
    src/common/metrics.c)
target_link_libraries(common PUBLIC ${M_LIB})

add_executable(gennoise src/app/main_gennoise.c)
target_link_libraries(gennoise common)

add_executable(imgcmp src/app/main_imgcmp.c)
target_link_libraries(imgcmp common)

# Phase-06 analysis tool: counts the corrupted pixels whose 3x3 window has no clean
# pixel left, i.e. how often the median fallback in core.h is actually taken.
add_executable(fbcount src/tools/fallback_count.c)
target_link_libraries(fbcount common)

# One main(), one implementation linked in: the only difference is DENOISE_FN.
add_executable(denoise_seq src/app/main_denoise.c src/seq/denoise_seq.c)
target_link_libraries(denoise_seq common)
target_compile_definitions(denoise_seq PRIVATE DENOISE_FN=denoise_seq_impl IMPL_NAME="seq")

# Phase 02 adds src/omp/denoise_omp.c; until then the target does not exist.
if(OpenMP_C_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/src/omp/denoise_omp.c)
    add_executable(denoise_omp src/app/main_denoise.c src/omp/denoise_omp.c)
    target_link_libraries(denoise_omp common OpenMP::OpenMP_C)
    target_compile_definitions(denoise_omp PRIVATE DENOISE_FN=denoise_omp_impl IMPL_NAME="omp"
                                                DENOISE_HAS_OMP=1)
endif()

if(ENABLE_CUDA)
    add_subdirectory(src/cuda)
endif()

# make check -> correctness pipeline for whatever has been built so far.
set(CHECK_BINS denoise_seq gennoise imgcmp)
if(TARGET denoise_omp)
    list(APPEND CHECK_BINS denoise_omp)
endif()
if(TARGET denoise_cuda)
    list(APPEND CHECK_BINS denoise_cuda)
endif()

add_custom_target(check
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/check.sh ${CMAKE_SOURCE_DIR} $<TARGET_FILE_DIR:denoise_seq>
    DEPENDS ${CHECK_BINS}
    USES_TERMINAL
    VERBATIM)
