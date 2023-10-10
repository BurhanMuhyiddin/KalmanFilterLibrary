cmake_minimum_required(VERSION 3.10)

project(KalmanFilter)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

include_directories(include)

set(SOURCE_FILES
    # kalman_filter.cpp
)

link_libraries(stdc++fs)

add_executable(${PROJECT_NAME} ekf_test.cpp ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIRS})