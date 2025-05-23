cmake_minimum_required(VERSION 3.14)
project(USRPAdvancedScanner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost 1.65 REQUIRED COMPONENTS program_options system thread)
find_package(UHD REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED)
find_package(FFTW3F REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/hdr
    ${Boost_INCLUDE_DIRS}
    ${UHD_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${FFTW3F_INCLUDE_DIRS}
)

set(SOURCES
    src/main.cpp
    src/config.cpp
    src/globals.cpp
    src/shared_data.cpp
    src/fft_processor.cpp
    src/ml_processor.cpp
    src/usrp_utils.cpp
    src/usrp_scanner.cpp
)

add_executable(usrp_scanner ${SOURCES})

target_link_libraries(usrp_scanner PRIVATE
    ${Boost_LIBRARIES}
    ${UHD_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${FFTW3F_LIBRARIES}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml")
    add_custom_command(TARGET usrp_scanner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(usrp_scanner PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

target_compile_options(usrp_scanner PRIVATE
    $<$<CONFIG:Debug>:-g -DDEBUG>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()