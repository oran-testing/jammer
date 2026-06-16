cmake_minimum_required(VERSION 3.16)

project(rtu_jammer
  VERSION 1.0
  DESCRIPTION "Basic Jammer utilizing UHD"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# ---------------------------------------------------------------------------
# Fetch rt-recon-sdk
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
  rt-recon-sdk
  # GIT_REPOSITORY https://github.com/cueltschey/rt-recon-sdk.git
  SOURCE_DIR /home/niraj-gupta/Documents/wireless/rt-recon-sdk
  # GIT_TAG        main
  # GIT_SHALLOW    TRUE
)

message(STATUS "Fetching rt-recon-sdk from GitHub...")
FetchContent_MakeAvailable(rt-recon-sdk)

# ---------------------------------------------------------------------------
# Existing dependencies
# ---------------------------------------------------------------------------
find_package(UHD)
if(UHD_FOUND)
  include_directories(${UHD_INCLUDE_DIRS})
  link_directories(${UHD_LIBRARY_DIRS})
  message(STATUS "UHD library found")
else()
  message(FATAL_ERROR "UHD library not found")
endif()

find_package(YAMLCPP)
if(YAMLCPP_FOUND)
  include_directories(${YAMLCPP_INCLUDE_DIR})
else()
  message(FATAL_ERROR "yaml-cpp is required to build ${CMAKE_PROJECT_NAME}")
endif()

# ---------------------------------------------------------------------------
# srsRAN 4G libs needed by rt-recon-autoconfig
# (srsran_phch does not exist as a separate lib — merged into srsran_phy)
# ---------------------------------------------------------------------------
find_library(SRSRAN_PHY_LIB    NAMES srsran_phy    REQUIRED)
find_library(SRSRAN_COMMON_LIB NAMES srsran_common  REQUIRED)
find_library(SUPPORT_LIB       NAMES support        REQUIRED)
find_library(SRSLOG_LIB        NAMES srslog         REQUIRED)
find_library(SRSRAN_RF_LIB     NAMES srsran_rf      REQUIRED)

message(STATUS "SRSRAN_PHY_LIB:    ${SRSRAN_PHY_LIB}")
message(STATUS "SRSRAN_COMMON_LIB: ${SRSRAN_COMMON_LIB}")
message(STATUS "SUPPORT_LIB:       ${SUPPORT_LIB}")
message(STATUS "SRSLOG_LIB:        ${SRSLOG_LIB}")
message(STATUS "SRSRAN_RF_LIB:     ${SRSRAN_RF_LIB}")

# ---------------------------------------------------------------------------
# Build jammer
# ---------------------------------------------------------------------------
file(GLOB JAMMER_SRC src/*.cc)

include_directories(hdr)
add_executable(jammer ${JAMMER_SRC})

target_link_libraries(jammer
  PRIVATE
    ${UHD_LIBRARIES}
    yaml-cpp
    rt-recon-autoconfig
    ${SRSRAN_PHY_LIB}
    ${SRSRAN_COMMON_LIB}
    ${SUPPORT_LIB}
    ${SRSLOG_LIB}
    ${SRSRAN_RF_LIB}
    pthread
)

install(
  PROGRAMS
  ${CMAKE_BINARY_DIR}/jammer
  DESTINATION /usr/local/bin/
)
