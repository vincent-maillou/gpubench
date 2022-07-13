# ----------------------------------------
# Liger's CMakeList for Kokkos compilation
# ----------------------------------------

cmake_minimum_required(VERSION 3.10)
project(kokkos_gpubench)

cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_COMPILER /usr/local/bin/nvcc_wrapper)
find_package(Kokkos REQUIRED)

add_executable(mesh_kokkos mesh_kokkos.cpp)
add_executable(dualview_dotp dualview_dotp.cpp)

target_link_libraries(mesh_kokkos Kokkos::kokkos)
target_link_libraries(dualview_dotp Kokkos::kokkos)
