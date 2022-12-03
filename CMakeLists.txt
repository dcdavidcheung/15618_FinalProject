cmake_minimum_required(VERSION 3.19)
project(dirt)

add_subdirectory(ext ext_build)

## include directories
include_directories(${PROJECT_SOURCE_DIR}/src
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/ext/tinyformat
                    ${PROJECT_SOURCE_DIR}/ext/json
                    ${PROJECT_SOURCE_DIR}/ext/stb_image
                    ${PROJECT_SOURCE_DIR}/ext/rang/include
                    ${PROJECT_SOURCE_DIR}/ext/filesystem
                    ${PROJECT_SOURCE_DIR}/ext/pcg32
)

set(dirt_headers
    include/dirt/ao.h
    include/dirt/argparse.h
    include/dirt/array2d.h
    include/dirt/bbh.h
    include/dirt/box.h
    include/dirt/camera.h
    include/dirt/common.h
    include/dirt/fwd.h
    include/dirt/image.h
    include/dirt/integrator.h
    include/dirt/material.h
    include/dirt/medium.h
    include/dirt/mesh.h
    include/dirt/normals.h
    include/dirt/obj.h
    include/dirt/onb.h
    include/dirt/parser.h
    include/dirt/path_tracer_simple.h
    include/dirt/path_tracer_mats.h
    include/dirt/path_tracer_mixture.h
    include/dirt/path_tracer_nee.h
    include/dirt/path_tracer_mis.h
    include/dirt/volpath_tracer_nee.h
    include/dirt/volpath_tracer_uni.h
    include/dirt/perlin.h
    include/dirt/primetable.h
    include/dirt/progress.h
    include/dirt/quad.h
    include/dirt/ray.h
    include/dirt/sampler.h
    include/dirt/scene.h
    include/dirt/sphere.h
    include/dirt/surface.h
    include/dirt/surfacegroup.h
    include/dirt/texture.h
    include/dirt/timer.h
    include/dirt/transform.h
    include/dirt/vec.h
	include/dirt/original.h
)

set(dirt_srcs
    src/argparse.cpp
    src/bbh.cpp
    src/common.cpp
    src/image.cpp
    src/integrator.cpp
    src/material.cpp
    src/medium.cpp
    src/mesh.cpp
    src/obj.cpp
    src/parser.cpp
    src/perlin.cpp
    src/progress.cpp
    src/quad.cpp
    src/sampler.cpp
    src/scene.cpp
    src/sphere.cpp
    src/surface.cpp
    src/surfacegroup.cpp
    src/testscenes.cpp
    src/texture.cpp
)

# Compile core dirt library
add_library(dirt_lib STATIC
    ${dirt_headers}
    ${dirt_srcs}
)

add_executable(dirt src/01_dirt.cpp)

add_executable(03_sample_test src/03_sample_test.cpp)
add_executable(03_point_gen src/03_point_gen.cpp)
add_executable(04_sample_tester src/04_sample_tester.cpp)
add_executable(05_phase_tester src/05_phase_tester.cpp)

target_link_libraries(dirt dirt_lib)
target_link_libraries(03_sample_test dirt_lib)
target_link_libraries(03_point_gen dirt_lib)
target_link_libraries(04_sample_tester dirt_lib)
target_link_libraries(05_phase_tester dirt_lib)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dirt_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

SOURCE_GROUP("ext\\tinyformat" FILES ${ext_tinyformat_srcs})
SOURCE_GROUP("ext\\json" FILES ${ext_json_srcs})
SOURCE_GROUP("ext\\stb_image" FILES ${ext_stb_image_srcs})
SOURCE_GROUP("ext\\rang" FILES ${ext_rang_srcs})
SOURCE_GROUP("ext\\filesystem" FILES ${ext_filesystem_srcs})
SOURCE_GROUP("ext\\pcg32" FILES ${ext_pcg32_srcs})
SOURCE_GROUP("dirt\\header files" FILES ${dirt_headers})
SOURCE_GROUP("dirt\\source files" FILES ${dirt_srcs})
