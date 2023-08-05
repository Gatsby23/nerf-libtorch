target_sources(train PRIVATE train.cpp)
target_sources(val PRIVATE val.cpp)
add_subdirectory(./nerf)
