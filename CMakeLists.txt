cmake_minimum_required (VERSION 2.8.11)

project (NeuralNetwork)

set(EIGEN_DIR "" CACHE PATH "Path to Eigen root directory")

include_directories(${NeuralNetwork_SOURCE_DIR}/include ${EIGEN_DIR})

subdirs(src)

add_executable(main main.cpp)
target_link_libraries(main NeuralNetwork)

