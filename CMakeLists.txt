cmake_minimum_required(VERSION 3.14)
project(USRPAdvancedScanner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Dependencies ---
find_package(Boost 1.65 REQUIRED COMPONENTS program_options system thread)
find_package(UHD REQUIRED)

# For FFTW3 (single precision float)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f) # Note: library name might be just 'fftw3f'

# For yaml-cpp
# Option A: Try standard find_package first, it might work on some systems
find_package(yaml-cpp 0.6) # Make it not REQUIRED initially to see if pkg-config is needed
if(NOT YAML_CPP_FOUND)
    message(STATUS "yaml-cpp not found via CMake config, trying PkgConfig...")
    pkg_check_modules(YAML_CPP REQUIRED IMPORTED_TARGET yaml-cpp) # Common .pc file name
    if(YAML_CPP_FOUND)
        set(YAML_CPP_LIBRARIES PkgConfig::YAML_CPP) # Use the imported target
        set(YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIRS}) # PkgConfig sets this
    else()
        message(FATAL_ERROR "yaml-cpp not found via CMake config or PkgConfig. Please install libyaml-cpp-dev and ensure yaml-cpp.pc is available.")
    endif()
else()
    message(STATUS "Found yaml-cpp via CMake config: Version ${YAML_CPP_VERSION}")
    # If found by config, YAML_CPP_LIBRARIES and YAML_CPP_INCLUDE_DIRS should be set
    # Sometimes the target is yaml-cpp::yaml-cpp
    if(TARGET yaml-cpp::yaml-cpp)
      set(YAML_CPP_LIBRARIES yaml-cpp::yaml-cpp)
    elseif(NOT YAML_CPP_LIBRARIES AND YAML_CPP_LIBRARY) # Older CMake find modules might set YAML_CPP_LIBRARY
      set(YAML_CPP_LIBRARIES ${YAML_CPP_LIBRARY})
    endif()
endif()


# --- Include Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/hdr
    ${Boost_INCLUDE_DIRS}
    ${UHD_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}  # Will be set by find_package or pkg_check_modules
    ${FFTW3F_INCLUDE_DIRS}    # Will be set by pkg_check_modules
)

# --- Source Files ---
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

# --- Link Libraries ---
target_link_libraries(usrp_scanner PRIVATE
    ${Boost_LIBRARIES}
    ${UHD_LIBRARIES}
    ${YAML_CPP_LIBRARIES} # Will be set by find_package or PkgConfig::YAML_CPP
    PkgConfig::FFTW3F     # Use the imported target from pkg_check_modules
)

# --- The rest of your CMakeLists.txt ---
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