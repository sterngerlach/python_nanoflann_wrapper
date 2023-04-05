
# CMakeLists.txt

include(CMakeParseArguments)
include(ExternalProject)

cmake_minimum_required(VERSION 3.16)

project(knn_lib C CXX)

# Find the Eigen package
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")

# Find the Nanoflann library
add_subdirectory(3rdparty/nanoflann)
message(STATUS "Nanoflann source directory: ${nanoflann_SOURCE_DIR}")

# Find the Pybind11 library
add_subdirectory(3rdparty/pybind11)

# Add the wrapper library
pybind11_add_module(knn_lib ${PROJECT_SOURCE_DIR}/src/knn_lib.cpp
  ${PROJECT_SOURCE_DIR}/src/kd_tree_nanoflann.cpp)

target_include_directories(knn_lib PUBLIC ${nanoflann_SOURCE_DIR}/include)
target_link_libraries(knn_lib PUBLIC Eigen3::Eigen)

target_compile_options(knn_lib PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-pedantic -Wall -Wextra -Wno-psabi>
  $<$<CONFIG:Release>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-O0 -ggdb>)
target_compile_definitions(knn_lib PRIVATE
  VERSION_INFO=${VERSION_INFO})
target_compile_features(knn_lib PUBLIC cxx_std_17)

# Add test programs
add_executable(knn_test ${PROJECT_SOURCE_DIR}/tests/knn_test.cpp)
target_include_directories(knn_test PUBLIC ${nanoflann_SOURCE_DIR}/include)
target_compile_options(knn_test PUBLIC
  $<$<CXX_COMPILER_ID:GNU>:-pedantic -Wall -Wextra -Wno-psabi>
  $<$<CONFIG:Release>:-O3>
  $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-O0 -ggdb>)
target_compile_features(knn_test PUBLIC cxx_std_17)
