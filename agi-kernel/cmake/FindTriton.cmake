# CMake find-module for the OpenAI Triton Python package.
# Sets:
#   TRITON_FOUND
#   TRITON_PYTHON_EXECUTABLE
#   TRITON_VERSION

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import triton, sys; print(triton.__version__)"
    OUTPUT_VARIABLE TRITON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _triton_rc
    ERROR_QUIET
)

if(_triton_rc EQUAL 0)
    set(TRITON_FOUND TRUE)
    set(TRITON_PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    message(STATUS "Triton found: version ${TRITON_VERSION}")
else()
    set(TRITON_FOUND FALSE)
    message(WARNING "Triton not found — fused kernels will be disabled.")
endif()
