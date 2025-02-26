cmake_minimum_required(VERSION 3.10)

project(rtu_jammer
  VERSION 1.0
  DESCRIPTION "Basic Jammer utilizing UHD"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")


find_package(UHD)
if(UHD_FOUND)
  set(RF_FOUND true)
  include_directories(${UHD_INCLUDE_DIRS})
  link_directories(${UHD_LIBRARY_DIRS})
  message(STATUS "UHD library found")
else()
  message(FATAL_ERROR "UHD library not found")
endif()


find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

file(GLOB JAMMER_SRC src/*.cc)

include_directories(hdr)
add_executable(jammer ${JAMMER_SRC})

target_link_libraries(jammer ${UHD_LIBRARIES} ${YAML_CPP_LIBRARIES})

