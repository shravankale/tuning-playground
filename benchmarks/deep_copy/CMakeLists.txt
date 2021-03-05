set(EXE_NAME deep_copy)
set(N_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/benchmarks/deep_copy)

add_executable(${EXE_NAME}.exe ${EXE_NAME}.cpp)
target_link_libraries(${EXE_NAME}.exe TuningPlaygound Kokkos::kokkos)
target_include_directories(${EXE_NAME}.exe PRIVATE ${N_INCLUDE_DIR})
install(TARGETS ${EXE_NAME}.exe)
