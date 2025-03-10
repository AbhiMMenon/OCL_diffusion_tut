cmake_minimum_required (VERSION 3.5 FATAL_ERROR)

project(OCL_solver LANGUAGES CXX)
find_package(OpenCL REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lOpenCL -Wall -Werror")
set(source_dir "${PROJECT_SOURCE_DIR}/src")

add_executable (solver ${source_dir}/solver.cpp)
target_include_directories(solver PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(solver PRIVATE ${OpenCL_LIBRARIES})
add_compile_definitions(diffusion PRIVATE CL_SOURCE="${source_dir}/kernels.cl" 
    DATA_SRC="${PROJECT_SOURCE_DIR}/data/init.dat"
    OUT_DAT="${PROJECT_SOURCE_DIR}/data/out.dat"
    INIT_DAT="${PROJECT_SOURCE_DIR}/data/init1.dat"
    )
