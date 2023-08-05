set(BUILD_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Setup opencv")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif()

message(STATUS "Setup pytorch")
FetchContent_Declare(
    pytorch
    URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
    SOURCE_DIR pytorch
)
FetchContent_MakeAvailable(pytorch)
include(${BUILD_LIB_DIR}/pytorch/share/cmake/Torch/TorchConfig.cmake)
list(APPEND CMAKE_PREFIX_PATH ${BUILD_LIB_DIR}/pytorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${BUILD_LIB_DIR}/pytorch/include/torch/csrc/api/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${BUILD_LIB_DIR}/pytorch/include)