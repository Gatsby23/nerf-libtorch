set(SRC
    dataset.cpp
    logger.cpp
    nerf.cpp
    positional_encoding.cpp
    radiance_field.cpp
    ray.cpp
    stdafx.cpp
    volume_rendering.cpp
    utils/image.cpp
)
target_precompile_headers(${PROJECT_NAME} PUBLIC stdafx.h)
target_precompile_headers(train REUSE_FROM ${PROJECT_NAME})
target_precompile_headers(val REUSE_FROM ${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SRC})