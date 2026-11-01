cmake_minimum_required(VERSION 3.20)
project(oprel_llama_ssd LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DOPREL_BUILD_DLL)
add_definitions(-DNOMINMAX)

include_directories(include)

set(SOURCES
    src/direct_io.cpp
    src/quest.cpp
    src/llama_ssd_wrapper.cpp
)

add_library(oprel_llama_ssd SHARED ${SOURCES})

if(WIN32)
    target_link_libraries(oprel_llama_ssd PRIVATE kernel32)
endif()

# Set output directory to bin
set_target_properties(oprel_llama_ssd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
