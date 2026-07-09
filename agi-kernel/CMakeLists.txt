# CortexOS build integration — CUDA + Triton + C++20.
#
# Design notes:
#   * We enable CUDA as a first-class language so CMake handles .cu compilation.
#   * Triton kernels are AOT-compiled into a shared library via a custom
#     command that invokes triton.compile() on each .py file.
#   * gtest is fetched via FetchContent if AGI_ENABLE_TESTS is on.

cmake_minimum_required(VERSION 3.24)
project(cortexos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AGI_ENABLE_CUDA    "Enable CUDA device kernels"   ON)
option(AGI_ENABLE_TRITON  "Enable Triton fused kernels"  ON)
option(AGI_ENABLE_TESTS   "Build gtest suite"            ON)
option(AGI_ENABLE_BENCHES "Build benchmarks"             ON)

if(AGI_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;90")   # A100 + H100 by default
    endif()
endif()

# ── Core library ────────────────────────────────────────────────────────────
set(AGI_HOST_SOURCES
    src/host/column.cpp
    src/host/connectivity.cpp
    src/host/neuromod.cpp
    src/host/thalamus.cpp
    src/host/scheduler.cpp
    src/host/runtime.cpp
    src/plasticity/plasticity.cpp
    src/memory/codecs.cpp
    src/io/sensor_encoders.cpp
    src/ir/ir_stub.cpp
)

set(AGI_CUDA_SOURCES
    src/cuda/hh_step_kernel.cu
    src/cuda/izh_step_kernel.cu
    src/cuda/csr_spmv_spike.cu
    src/cuda/aer_route.cu
    src/cuda/stdp_update.cu
    src/cuda/thalamus_route.cu
    src/cuda/wta_select.cu
    src/cuda/replay.cu
    src/cuda/neuromod_broadcast.cu
)

if(AGI_ENABLE_CUDA)
    add_library(cortexos STATIC ${AGI_HOST_SOURCES} ${AGI_CUDA_SOURCES})
    set_target_properties(cortexos PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(cortexos STATIC ${AGI_HOST_SOURCES})
endif()

target_include_directories(cortexos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(cortexos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mock_cuda)

# ── Triton AOT compilation ──────────────────────────────────────────────────
if(AGI_ENABLE_TRITON)
    set(AGI_TRITON_KERNELS
        src/triton/column_step.py
        src/triton/cable_step.py
        src/triton/stdp_fused.py
    )
    set(TRITON_CACHE ${CMAKE_BINARY_DIR}/triton_cache)
    file(MAKE_DIRECTORY ${TRITON_CACHE})
    foreach(kern ${AGI_TRITON_KERNELS})
        get_filename_component(name ${kern} NAME_WE)
        add_custom_command(
            OUTPUT  ${TRITON_CACHE}/${name}.so
            COMMAND python -m triton.tools.compile ${kern} --output ${TRITON_CACHE}/${name}.so
            DEPENDS ${kern}
            COMMENT "AOT-compiling Triton kernel ${name}"
        )
        list(APPEND TRITON_SOS ${TRITON_CACHE}/${name}.so)
    endforeach()
    add_custom_target(triton_kernels ALL DEPENDS ${TRITON_SOS})
    add_dependencies(cortexos triton_kernels)
endif()

# ── Examples ────────────────────────────────────────────────────────────────
add_executable(example_single_column    examples/single_column.cpp)
target_link_libraries(example_single_column   PRIVATE cortexos)

add_executable(example_dopamine_learning examples/dopamine_learning.cpp)
target_link_libraries(example_dopamine_learning PRIVATE cortexos)

add_executable(example_attention_task    examples/attention_task.cpp)
target_link_libraries(example_attention_task PRIVATE cortexos)

# ── Tests ───────────────────────────────────────────────────────────────────
if(AGI_ENABLE_TESTS)
    enable_testing()
    add_executable(test_hh_dynamics tests/test_hh_dynamics.cpp)
    target_link_libraries(test_hh_dynamics PRIVATE cortexos)
    add_test(NAME hh_dynamics COMMAND test_hh_dynamics)

    add_executable(test_compression_roundtrip tests/test_compression_roundtrip.cpp)
    target_link_libraries(test_compression_roundtrip PRIVATE cortexos)
    add_test(NAME compression_roundtrip COMMAND test_compression_roundtrip)

    add_executable(test_csr_spmv tests/test_csr_spmv.cpp)
    target_link_libraries(test_csr_spmv PRIVATE cortexos)
    add_test(NAME csr_spmv COMMAND test_csr_spmv)
endif()

# ── Benchmarks ──────────────────────────────────────────────────────────────
if(AGI_ENABLE_BENCHES)
    add_executable(bench_scaling benchmarks/bench_scaling.cpp)
    target_link_libraries(bench_scaling PRIVATE cortexos)
endif()

install(TARGETS cortexos
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/agi DESTINATION include)
