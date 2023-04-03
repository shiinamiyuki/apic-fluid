cmake_minimum_required(VERSION 3.0)
project(apic_fluid_cpp_extra)
set(LIBIGL_OPENGL ON CACHE BOOL "Enable igl::opengl")
set(LIBIGL_GLFW ON CACHE BOOL "Enable igl::glfw")
add_subdirectory(libigl)
add_library(cpp_extra SHARED cpp_extra.cpp)

target_link_libraries(cpp_extra PUBLIC igl::core igl::glfw)
