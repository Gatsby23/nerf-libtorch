# cmake settings
cmake_minimum_required(VERSION 3.27)
enable_language(CXX)
set(CMAKE_C_STANDARD    17)
set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_HOME /usr/local/cuda-12.2/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# project settings
project(nerf LANGUAGES CXX C CUDA)
if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED True)

# add executable
add_executable(train)
add_executable(val)
add_library(${PROJECT_NAME})
add_subdirectory(./src)
# target_compile_definitions(${PROJECT_NAME} PUBLIC _DEBUG)
# target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DISTRIBUTED)

# linking
target_link_libraries(train PRIVATE ${PROJECT_NAME})
target_link_libraries(val PRIVATE ${PROJECT_NAME})

# add libralies
include(./cmake/dependencies.cmake)