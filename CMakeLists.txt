cmake_minimum_required(VERSION 3.5)

project(mpi_bellman_ford)

set (CMAKE_CXX_COMPILER mpic++)
set (CMAKE_EXPORT_COMPILE_COMMANDS on)

#including mpi
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set (source_files mpi_bellman_ford.cpp)

add_executable(mpi_bellman_ford ${source_files})
# target_link_libraries(mpi_bellman_ford PUBLIC MPI::MPI_CXX)
