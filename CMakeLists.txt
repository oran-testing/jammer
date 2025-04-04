cmake_minimum_required(VERSION 3.10)

project(rtu_jammer
  VERSION 1.0
  DESCRIPTION "Basic Jammer utilizing UHD"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # Recommended

# Add local modules path (for FindYAMLCPP.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# --- Find Required Packages ---

# Find UHD (Uses older variables based on previous errors)
find_package(UHD REQUIRED)
if(UHD_FOUND)
  include_directories(${UHD_INCLUDE_DIRS})
  # link_directories(${UHD_LIBRARY_DIRS}) # Avoid link_directories; prefer linking the variable
  message(STATUS "UHD library found: ${UHD_LIBRARIES}")
  message(STATUS "UHD includes found: ${UHD_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "UHD library not found")
endif()

# Find Boost (Program Options component is required)
find_package(Boost REQUIRED COMPONENTS program_options)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS}) # Add Boost includes
  message(STATUS "Boost library found")
  message(STATUS "Boost Includes: ${Boost_INCLUDE_DIRS}")
  message(STATUS "Boost Program Options Library: ${Boost_PROGRAM_OPTIONS_LIBRARY}")
else()
    message(FATAL_ERROR "Boost (program_options) library not found")
endif()

# Find FFTW3 Float using PkgConfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f) # Finds fftw3f using pkg-config
if(FFTW3F_FOUND)
    include_directories(${FFTW3F_INCLUDE_DIRS}) # Add FFTW includes
    message(STATUS "FFTW3 Float library found")
    message(STATUS "FFTW3F Includes: ${FFTW3F_INCLUDE_DIRS}")
    message(STATUS "FFTW3F Linker Flags: ${FFTW3F_LIBRARIES}")
else()
    message(FATAL_ERROR "FFTW3 Float (fftw3f) library not found via pkg-config")
endif()

# Find YAMLCPP (uses custom FindYAMLCPP.cmake from cmake/modules)
find_package(YAMLCPP REQUIRED)
if (YAMLCPP_FOUND)
  include_directories(${YAMLCPP_INCLUDE_DIR})
  # link_directories(${YAMLCPP_LIBRARY}) # WRONG! link_directories takes DIRECTORIES, not FILES. Remove.
  message(STATUS "Found YAMLCPP: ${YAMLCPP_LIBRARY}")
  message(STATUS "YAMLCPP Includes: ${YAMLCPP_INCLUDE_DIR}")
else (YAMLCPP_FOUND)
  message(FATAL_ERROR "yaml-cpp is required to build ${CMAKE_PROJECT_NAME}")
endif (YAMLCPP_FOUND)

# --- Define Source Files ---
# Using GLOB is convenient but less robust than listing files explicitly.
# If you add/remove .cc files, you must re-run cmake.
# Consider replacing with: set(JAMMER_SRC src/datacollection.cc src/otherfile.cc)
file(GLOB JAMMER_SRC src/*.cc)

# Add local header directory
include_directories(hdr)

# Define the executable target
add_executable(jammer ${JAMMER_SRC})

# --- Link Libraries to the Executable ---
# Linking using the variables populated by the find_package calls above

target_link_libraries(jammer PRIVATE
    # UHD Library - Use the variable found by find_package(UHD)
    ${UHD_LIBRARIES}

    # Boost Program Options - Use the specific library variable
    ${Boost_PROGRAM_OPTIONS_LIBRARY}

    # FFTW3 Float Library - Use the variable populated by pkg_check_modules
    ${FFTW3F_LIBRARIES}

    # yaml-cpp Library - Use the full path variable found by FindYAMLCPP.cmake
    ${YAMLCPP_LIBRARY}

    # Link Boost::system explicitly if needed (sometimes a dependency of program_options)
    # Check find_package(Boost ...) documentation if system is automatically found/linked.
    # If you get errors related to Boost.System, you might need find_package(... COMPONENTS system)
    # and link ${Boost_SYSTEM_LIBRARY}
    # ${Boost_SYSTEM_LIBRARY}

    # Link math library explicitly (-lm)
    m

    # Link pthreads explicitly if needed
    # find_package(Threads) # Usually found by Boost
    # Threads::Threads
)

# --- Optional: Compiler Warnings (Recommended) ---
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
  target_compile_options(jammer PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    # Add other warnings as needed
  )
endif()