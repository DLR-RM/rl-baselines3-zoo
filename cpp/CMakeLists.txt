cmake_minimum_required(VERSION 3.16.3)
project(baselines3_models)
include(cmake/CMakeRC.cmake)

cmrc_add_resource_library(baselines3_model_resources
ALIAS baselines3_model::rc
NAMESPACE baselines3_model

#static
#!static
)

# Install PyTorch C++ first, see: https://pytorch.org/cppdocs/installing.html
# Don't forget to add it to your CMAKE_PREFIX_PATH
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#Enable C++17
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -fPIC")

option(BASELINES3_BIN "Building bin" OFF)
option(BASELINES3_PYBIND "Build python bindings (requires pybind11)" OFF)

set(ALL_SOURCES
src/baselines3_models/predictor.cpp
src/baselines3_models/preprocessing.cpp

#sources
#!sources
)

set(LIBRARIES
"${TORCH_LIBRARIES}" baselines3_model::rc
)

add_library(baselines3_models SHARED ${ALL_SOURCES})
target_link_libraries(baselines3_models ${LIBRARIES})
target_include_directories(baselines3_models PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if (BASELINES3_BIN)
    add_executable(predict ${CMAKE_CURRENT_SOURCE_DIR}/src/predict.cpp)
    target_link_libraries(predict baselines3_models)
endif()

if (BASELINES3_PYBIND)
    set (Python_EXECUTABLE "/usr/bin/python3.8")
    # apt-get install python3-dev
    find_package(Python COMPONENTS Interpreter Development)
    # apt-get install python3-pybind11
    find_package(pybind11 REQUIRED)

    pybind11_add_module(baselines3_py ${ALL_SOURCES})
    target_link_libraries(baselines3_py PRIVATE ${LIBRARIES})
    target_compile_definitions(baselines3_py PUBLIC -DEXPORT_PYBIND)

    target_include_directories(baselines3_py PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
endif()