cmake_minimum_required(VERSION 3.25)
project(spark_mlx_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_VARIABLE MLX_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(MLX CONFIG REQUIRED)

if(APPLE)
  # Official wheels can contain framework paths from the build machine's SDK.
  foreach(framework Metal Foundation QuartzCore Accelerate)
    find_library(${framework}_LIBRARY ${framework} REQUIRED)
    list(APPEND spark_frameworks "${${framework}_LIBRARY}")
  endforeach()
  set_target_properties(mlx PROPERTIES INTERFACE_LINK_LIBRARIES "${spark_frameworks}")
endif()

nanobind_add_module(
  _ext NB_STATIC FREE_THREADED LTO NOMINSIZE NB_DOMAIN mlx
  src/bindings.cpp src/spark_qkv_rope.cpp src/spark_w8a8.cpp src/spark_w8a16.cpp)
target_link_libraries(_ext PRIVATE mlx)

mlx_build_metallib(
  TARGET spark_w8a8_kernels
  TITLE spark_w8a8
  SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/w8a8_matmul.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/w8a8_tensor.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/w8a8_qmv.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/w8a8_quantize.metal"
  INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
  COMPILE_OPTIONS -std=metal4.0
  OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
add_dependencies(_ext spark_w8a8_kernels)

mlx_build_metallib(
  TARGET spark_w8a16_kernels
  TITLE spark_w8a16
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/w8a16_matvec.metal"
  INCLUDE_DIRS ${MLX_INCLUDE_DIRS}
  COMPILE_OPTIONS -std=metal4.0
  OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
add_dependencies(_ext spark_w8a16_kernels)
