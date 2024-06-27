cmake_minimum_required(VERSION 3.10)

enable_testing()

# find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(hellocuda LANGUAGES CXX CUDA)

# Add the CUDA source files to the project
add_executable(main ./test/test_gemm.cu)

# Add the source directory to the include directories
target_include_directories(main PUBLIC ./include ./src)

#link cublas
target_link_libraries(main ${CUDA_LIBRARIES} cublas)

# CUDA compile options
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)


# 定义测试文件列表
set(test_files
    "test_gemm"
    "test_relu"
)

# Loop to add tests
foreach(test_file ${test_files})
    # Add executable for each test file
    add_executable(${test_file} ./test/${test_file}.cu)

    # Add include directories for each test
    target_include_directories(${test_file} PUBLIC ./include ./src)

    # Link cuBLAS library and CUDA runtime library for each test
    target_link_libraries(${test_file} ${CUDA_LIBRARIES} cublas)

    # CUDA compile options for each test
    target_compile_options(${test_file} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_options(${test_file} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    # Add CMake test for each executable
    add_test(NAME ${test_file} COMMAND ${test_file})
endforeach()

set(CMAKE_CUDA_ARCHITECTURES "sm_60;sm_70;sm_75;sm_80")