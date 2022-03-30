cmake_minimum_required(VERSION 3.16.3)
project(baselines3_models)
include(cmake/CMakeRC.cmake)

cmrc_add_resource_library(baselines3_model_resources
        ALIAS baselines3_model::rc
        NAMESPACE baselines3_model
        assets/approach_v0_model.pt
        )

# Install PyTorch C++ first, see: https://pytorch.org/cppdocs/installing.html
# Don't forget to add it to your CMAKE_PREFIX_PATH
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#Enable C++17
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -fPIC")

option(BASELINES3_BIN "Building bin" OFF)

set(ALL_SOURCES
    src/baselines3_models/predictor.cpp
    src/baselines3_models/preprocessing.cpp
    src/baselines3_models/approach_v0.cpp
)

add_library(baselines3_models SHARED ${ALL_SOURCES})
target_link_libraries(baselines3_models "${TORCH_LIBRARIES}" baselines3_model::rc)
target_include_directories(baselines3_models PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if (BASELINES3_BIN)
    add_executable(predict ${CMAKE_CURRENT_SOURCE_DIR}/src/predict.cpp)
    target_link_libraries(predict baselines3_models)
endif()