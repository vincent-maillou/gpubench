cmake_minimum_required(VERSION 3.10)
project(raja_benchmark LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(Threads)
find_package(RAJA REQUIRED)

set_source_files_properties(init_raja.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(dotp_raja.cpp PROPERTIES LANGUAGE CUDA)
add_compile_options(-arch sm_80 --expt-extended-lambda --expt-relaxed-constexpr)

add_executable(init_raja init_raja.cpp)
add_executable(dotp_raja dotp_raja.cpp)

target_link_libraries(init_raja PUBLIC RAJA)
target_link_libraries(dotp_raja PUBLIC RAJA)
