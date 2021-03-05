# this project adapted from ArborX (git@github.com:arborx/arborx)
cmake_minimum_required(VERSION 3.12)
project(TuningPlayground CXX)

find_package(Kokkos 3.1 REQUIRED)
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

add_library(TuningPlayground INTERFACE)
target_link_libraries(TuningPlayground INTERFACE Kokkos::kokkos)
set_target_properties(TuningPlayground PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_14)

include(CMakeDependentOption)
cmake_dependent_option(TUNINGPLAYGROUND_ENABLE_ROCTHRUST "Enable rocThrust support" ON "Kokkos_ENABLE_HIP" OFF)
if(Kokkos_ENABLE_HIP AND TUNINGPLAYGROUND_ENABLE_ROCTHRUST)
  # Require at least rocThrust-2.10.5 (that comes with ROCm 3.9) because
  # rocPRIM dependency is not set properly in exported configuration for
  # earlier versions
  find_package(rocthrust 2.10.5 REQUIRED CONFIG)
  target_link_libraries(TuningPlayground INTERFACE roc::rocthrust)
endif()

if(Kokkos_ENABLE_HIP AND NOT TUNINGPLAYGROUND_ENABLE_ROCTHRUST)
  message(WARNING "rocThrust is NOT enabled.\nThis will negatively impact performance on AMD GPUs.")
endif()
#DEBUG
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# Refer to the alias target in examples and benchmarks so they can be built
# against an installed TuningPlayground
add_library(TuningPlayground::TuningPlayground ALIAS TuningPlayground)

target_include_directories(TuningPlayground INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/details>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/details>
)

install(TARGETS TuningPlayground
  EXPORT TuningPlaygroundTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
)

install(EXPORT TuningPlaygroundTargets
  NAMESPACE TuningPlayground
  DESTINATION lib/cmake/TuningPlayground
)

include(CMakePackageConfigHelpers)


install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION include
        FILES_MATCHING PATTERN "*.hpp")

if (NOT CMAKE_BUILD_TYPE)
  set(default_build_type "RelWithDebInfo")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

option(TUNINGPLAYGROUND_ENABLE_BENCHMARKS "Enable benchmarks" OFF)

add_subdirectory(benchmarks)
