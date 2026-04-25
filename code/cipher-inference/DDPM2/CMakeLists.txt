cmake_minimum_required(VERSION 3.16)
project(shark)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(ext/cryptoTools)

add_library(${PROJECT_NAME}
    src/utils/timer.cpp
    src/utils/comm.cpp
    src/protocols/common.cpp
    src/protocols/init.cpp
    src/protocols/finalize.cpp
    src/protocols/input.cpp
    src/protocols/output.cpp
    src/protocols/matmul.cpp
    src/protocols/relutruncate.cpp
    src/protocols/drelu.cpp
    src/protocols/select.cpp
    src/protocols/relu.cpp
    src/protocols/conv.cpp
    src/protocols/maxpool.cpp
    src/protocols/spline.cpp
    src/protocols/lrs.cpp
    src/protocols/lrs_exact.cpp
    src/protocols/sumpool.cpp
    src/protocols/mul.cpp
    src/protocols/lut.cpp
    src/protocols/reciprocal.cpp
    src/protocols/softmax.cpp
    src/crypto/dcfbit.cpp
    src/crypto/dcfring.cpp
    src/crypto/dpfring.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen cryptoTools)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/clip.cpp)
  add_executable(benchmark-clip benchmarks/clip.cpp)
  target_link_libraries(benchmark-clip ${PROJECT_NAME})
else()
  message(WARNING "benchmarks/clip.cpp not found; skipping benchmark-clip target")
endif()

add_executable(benchmark-unclip_img2img benchmarks/unclip_img2imgsmall.cpp)
target_link_libraries(benchmark-unclip_img2img ${PROJECT_NAME})

# Keep a target name that matches the file name used in docs/scripts.
add_executable(benchmark-unclip_img2imgsmall benchmarks/unclip_img2imgsmall.cpp)
target_link_libraries(benchmark-unclip_img2imgsmall ${PROJECT_NAME})

add_executable(benchmark-ddpm benchmarks/ddpm.cpp)
target_link_libraries(benchmark-ddpm ${PROJECT_NAME})
