# Only reached with -DENABLE_CUDA=ON, so a box without nvcc never parses this file.
enable_language(CUDA)

# -arch=native needs CUDA >= 11.5. Older toolkit: leave CMAKE_CUDA_ARCHITECTURES
# unset and let CMake pick its default list, which is always valid.
if(NOT CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The shared main is reused verbatim, so -d/-m/-n/-r/-q and the rep loop cannot
# drift between seq/omp/cuda. DENOISE_HAS_CUDA only adds the transfer/kernel split
# to the report line. No -use_fast_math: identical double maths to the CPU matters
# more than a few percent of speedup.
add_executable(denoise_cuda
    ${CMAKE_SOURCE_DIR}/src/app/main_denoise.c
    kernels.cu
    host.cu)
target_include_directories(denoise_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(denoise_cuda common)
target_compile_definitions(denoise_cuda PRIVATE
    DENOISE_FN=denoise_cuda_impl
    IMPL_NAME="cuda"
    DENOISE_HAS_CUDA=1)

# Everything lands in build/ next to denoise_seq/denoise_omp: docs, check.sh and
# the report all reference build/denoise_cuda.
set_target_properties(denoise_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
