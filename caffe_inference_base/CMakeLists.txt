cmake_minimum_required(VERSION 3.5)
project(facethink_caffe_inference)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -fPIC -ggdb")
if (NOT WIN32)
  SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fPIC")
  SET(CMAKE_CXX_FLAGS "-fPIC")
  SET(CMAKE_C_FLAGS "-fPIC")
endif()

option (CPU_ONLY "USE CPU_ONLY FOR CAFFE INFERENCE" OFF)
option (BUILD_TESTS "build test codes for caffe inference" ON)

if (WIN32)
  set(Boost_USE_STATIC_LIBS ON)  # static boost
endif()

include(cmake/CheckDependencies.cmake)
detect_all()

#if (NOT WIN32)
#  include(cmake/Protoc.cmake)
#  caffe_protobuf_generate_cpp(${PROTOBUF_COMPILER})
#endif()

set(PROJECT_INCLUDE  ${PROJECT_INCLUDE} ${PROJECT_SOURCE_DIR})
set(PROJECT_LIBRARY  ${PROJECT_LIBRARY})
include_directories(${PROJECT_INCLUDE})

if (NOT CPU_ONLY)
  set(PLATFORM_TYPE "GPU")
else()
  set(PLATFORM_TYPE "CPU")
  add_definitions(-DCPU_ONLY)
endif()

set(LIBRARY_NAME facethink_caffe_${PLATFORM_TYPE})

if (WIN32)
  list(APPEND CUDA_NVCC_FLAGS " -Xcompiler \"/wd 4819\"")
  list(APPEND CUDA_NVCC_FLAGS " -Xcompiler \"/MD\"")
  add_definitions(-DNO_STRICT)
  #add_definitions(-DBOOST_ALL_DYN_LINK)  #only for dynamic library
  add_definitions(-DDLL_EXPORTS) #only for dynamic library
else()
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -Wno-deprecated")
  add_definitions(-DBOOST_ALL_DYN_LINK)

endif()

add_subdirectory(./caffe)
if (BUILD_TESTS)
  add_subdirectory(./test)
endif()

set(FT_CAFFE_INCLUDE ${PROJECT_INCLUDE} CACHE INTERNAL "facethink caffe inference include Library location." )
set(FT_CAFFE_NAME ${LIBRARY_NAME}  CACHE INTERNAL "facethink caffe inference library name." )
set(FT_CAFFE_PLATFORM_TYPE ${PLATFORM_TYPE} CACHE INTERNAL "facethink caffe inference flatform type (CPU or GPU).")
