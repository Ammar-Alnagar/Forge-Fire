cmake_minimum_required(VERSION 3.18)
project(llm-engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
find_package(Protobuf REQUIRED)
find_package(OpenBLAS) # Or another BLAS library like Eigen

# --- Options ---
option(USE_OPENBLAS "Use OpenBLAS for GEMM" ON)
option(ENABLE_SIMD "Enable SIMD intrinsics" OFF)

# --- Protobuf ---
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx_proto)
set(PROTO_FILE ${PROTO_DIR}/onnx.proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

# This creates the target that generates the .pb.cc and .pb.h files
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

# --- Executable ---
add_executable(infer
    src/main.cpp
    src/loaders/onnx_loader.cpp
    ${PROTO_SRCS}
)

# --- Include directories ---
target_include_directories(infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR} # For generated headers
)

# --- Linking ---
target_link_libraries(infer PRIVATE
    ${Protobuf_LIBRARIES}
)

if(USE_OPENBLAS AND OpenBLAS_FOUND)
    target_link_libraries(infer PRIVATE ${OpenBLAS_LIBRARIES})
    target_compile_definitions(infer PRIVATE USE_OPENBLAS)
endif()

# --- SIMD specific flags (example for AVX2) ---
if(ENABLE_SIMD)
    # Add compiler flags for AVX2, guarded
    # This part will be expanded later
endif()

# --- Tests (to be added later) ---
enable_testing()
# add_subdirectory(tests)