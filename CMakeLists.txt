cmake_minimum_required(VERSION 3.10)

project(body_detect)
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/package")

######################################################################
#全局编译标志
######################################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

######################################################################
#全局链接和包含
######################################################################
include_directories(${PROJECT_SOURCE_DIR})
link_directories(${CMAKE_SOURCE_DIR}/third_party/opencv/lib)
link_directories(${CMAKE_SOURCE_DIR}/third_party/json/lib)

#################################################################L#####
#全局编译输出指定
######################################################################
#IF(CMAKE_BUILD_TYPE MATCHES Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
#ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

#string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${BUILD_TYPE})


######################################################################
#子工程目录
######################################################################
add_subdirectory(det_body_ssd)
add_subdirectory(caffe_inference_base)
######################################################################
#编译对象
######################################################################

add_executable(bodydetect
    body_detect.cpp)

target_include_directories(bodydetect PRIVATE
    ${PROJECT_SOURCE_DIR}/breakpad/src
    ${PROJECT_SOURCE_DIR}/third_party/json/include
    ${PROJECT_SOURCE_DIR}/third_party/opencv/include)

target_link_libraries(bodydetect
        det_body_ssd
        opencv_core
        jsoncpp
        )
add_dependencies(bodydetect facethink_caffe_GPU)
