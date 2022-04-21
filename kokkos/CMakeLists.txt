cmake_minimum_required(VERSION 3.10)
project(kokkos_gpubench)

cmake_policy(SET CMP0074 NEW)
#set(Kokkos_ROOT dont/know/not/needed)

set(CMAKE_CXX_COMPILER /usr/local/bin/nvcc_wrapper)
find_package(Kokkos REQUIRED)

add_executable(init_kokkos init_kokkos.cpp)
add_executable(init_kokkos_lambda init_kokkos_lambda.cpp)
add_executable(dotp_kokkos dotp_kokkos.cpp)

target_link_libraries(init_kokkos Kokkos::kokkos)
target_link_libraries(init_kokkos_lambda Kokkos::kokkos)
target_link_libraries(dotp_kokkos Kokkos::kokkos)


