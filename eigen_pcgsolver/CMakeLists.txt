cmake_minimum_required(VERSION 3.0)
add_subdirectory(eigen)

project(pcg_solver)

add_library(solve STATIC solve.cpp)
target_link_libraries(solve Eigen3::Eigen)