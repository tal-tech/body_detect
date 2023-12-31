#####################################################################################
function(detect_cuda)

  find_package(CUDA REQUIRED)
  if (CUDA_FOUND)
    set(HAVE_CUDA TRUE)
    set(CUDA_INCLUDE ${CUDA_INCLUDE_DIRS})
    set(CUDA_LIBRARY ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})

  endif()

  if (HAVE_CUDA)
    message(STATUS "Found CUDA: found (include: ${CUDA_INCLUDE}, library: ${CUDA_LIBRARY})")
    set(HAVE_CUDA ${HAVE_CUDA}  PARENT_SCOPE )
    set(CUDA_INCLUDE ${CUDA_INCLUDE} PARENT_SCOPE)
    set(CUDA_LIBRARY ${CUDA_LIBRARY} PARENT_SCOPE)
  endif()

endfunction()

################################################################################################

################################################################################################
# Short command for cuDNN detection. Believe it soon will be a part of CUDA toolkit distribution.
# That's why not FindcuDNN.cmake file, but just the macro
# Usage:
#   detect_cuDNN()
function(detect_cuDNN)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

  find_path(CUDNN_INCLUDE cudnn.h
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
    DOC "Path to cuDNN include directory." )

  # dynamic libs have different suffix in mac and linux
  if(APPLE)
    set(CUDNN_LIB_NAME "libcudnn.dylib")
  elseif(WIN32)
    set(CUDNN_LIB_NAME "cudnn.lib")
  else()
    set(CUDNN_LIB_NAME "libcudnn.so")
  endif()

  get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
  find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
    PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
    DOC "Path to cuDNN library.")

  if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
    set(HAVE_CUDNN  TRUE PARENT_SCOPE)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

    # cuDNN v3 and beyond
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
      CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
      CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
      CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
      CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
      CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
      CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

    if(NOT CUDNN_VERSION_MAJOR)
      set(CUDNN_VERSION "???")
    else()
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    endif()

    message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

    string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 6 cuDNNVersionIncompatible)
    if(cuDNNVersionIncompatible)
      message(FATAL_ERROR "cuDNN version >= 6 is required.")
    endif()

    set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

  endif()

endfunction()

########################################################################################
function(detect_protobuf)

  if (WIN32)
    set(HAVE_PROTOBUF TRUE)
     set(PROTOBUF_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/protobuf/2.6/include)
     file(GLOB PROTOBUF_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/protobuf/2.6/lib/win/v140/x64/Release/protobuf.lib)
     file(GLOB PROTOBUF_COMPILER ${PROJECT_SOURCE_DIR}/third_party/win/protobuf/bin/*.exe)
	
	set(PROTOBUF_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/protobuf/3.5.1/include)
    file(GLOB PROTOBUF_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/protobuf/3.5.1/lib/x64/libprotobuf.lib)
	
  else()
      set(HAVE_PROTOBUF TRUE)
      set(PROTOBUF_INCLUDE ${CMAKE_SOURCE_DIR}/grpc/include)
      set(PROTOBUF_LIBRARY libprotobuf)
      set(PROTOBUF_COMPILER $<TARGET_FILE:protoc>)
  endif()

  if (HAVE_PROTOBUF)
    message(STATUS "Found Protobuf: found (include: ${PROTOBUF_INCLUDE}, library: ${PROTOBUF_LIBRARY}), compiler: ${PROTOBUF_COMPILER}")
    set(HAVE_PROTOBUF ${HAVE_PROTOBUF}  PARENT_SCOPE )
    set(PROTOBUF_INCLUDE ${PROTOBUF_INCLUDE} PARENT_SCOPE)
    set(PROTOBUF_LIBRARY ${PROTOBUF_LIBRARY} PARENT_SCOPE)
    set(PROTOBUF_COMPILER ${PROTOBUF_COMPILER} CACHE INTERNAL "protobuf compiler location." )
  endif()

endfunction()

#########################################################################################
function(detect_boost)

  if (WIN32)
    set(HAVE_BOOST TRUE)
    #set(BOOST_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/boost/boost_1_62_0/include)
    #file(GLOB BOOST_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/boost/boost_1_62_0/libs/x64/*mt-1_62.lib)
	set(BOOST_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/boost/boost_1_66_0/include)
    file(GLOB BOOST_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/boost/boost_1_66_0/lib/x64/*.lib)
  else()
    find_package(Boost REQUIRED COMPONENTS filesystem log system)
    if (Boost_FOUND)
      set(HAVE_BOOST TRUE)
      set(BOOST_INCLUDE ${Boost_INCLUDE_DIRS})
      set(BOOST_LIBRARY ${Boost_LIBRARIES})
    endif()
    #set(HAVE_BOOST TRUE)
    #set(BOOST_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/boost/boost_1_62_0/include)
    #file(GLOB BOOST_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/boost/boost_1_62_0/libs/Linux/x64/gcc5.4/Release/*.a)
  endif()

  if (HAVE_BOOST)
    message(STATUS "Found Boost: found (include: ${BOOST_INCLUDE}, library: ${BOOST_LIBRARY})")
    set(HAVE_BOOST ${HAVE_BOOST} PARENT_SCOPE)
    set(BOOST_INCLUDE ${BOOST_INCLUDE} PARENT_SCOPE)
    set(BOOST_LIBRARY ${BOOST_LIBRARY} PARENT_SCOPE)
  endif()

endfunction()


#########################################################################################
function(detect_tbb)

  if (WIN32)
    set(HAVE_TBB TRUE)
    set(TBB_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/tbb/include)
    file(GLOB TBB_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/tbb/lib/x64/v140/Release/*.lib)
  else()
    # only required by boost-win.
  endif()

  if (HAVE_TBB)
    message(STATUS "Found TBB: found (include: ${TBB_INCLUDE}, library: ${TBB_LIBRARY})")
    set(HAVE_TBB ${HAVE_TBB} PARENT_SCOPE)
    set(TBB_INCLUDE ${TBB_INCLUDE} PARENT_SCOPE)
    set(TBB_LIBRARY ${TBB_LIBRARY} PARENT_SCOPE)
  endif()

endfunction()

#########################################################################################
function(detect_opencv)

  if (WIN32)
    set(HAVE_OPENCV TRUE)
    set(OPENCV_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/opencv/include)
    file(GLOB OPENCV_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/opencv/lib/x64/v140/Release/*.lib)
  else()
    set(HAVE_OPENCV TRUE)
    set(OPENCV_INCLUDE ${CMAKE_SOURCE_DIR}/opencv3.4.5/include/)
    file(GLOB OPENCV_LIBRARY ${CMAKE_SOURCE_DIR}/opencv3.4.5/lib/*)
  endif()

  if (HAVE_OPENCV)
    message(STATUS "Found Opencv: found (include: ${OPENCV_INCLUDE}, library: ${OPENCV_LIBRARY})")
    set(HAVE_OPENCV ${HAVE_OPENCV}  PARENT_SCOPE )
    set(OPENCV_INCLUDE ${OPENCV_INCLUDE} PARENT_SCOPE)
    set(OPENCV_LIBRARY ${OPENCV_LIBRARY} PARENT_SCOPE)
  endif()

endfunction()

###########################################################################################
function(detect_blas)

  if (WIN32)
    set(HAVE_BLAS TRUE)
    set(BLAS_INCLUDE ${PROJECT_SOURCE_DIR}/../3rdParty/openblas/0.2.19/Win64/include)
    file(GLOB BLAS_LIBRARY ${PROJECT_SOURCE_DIR}/../3rdParty/openblas/0.2.19/Win64/lib/libopenblas.dll.a)
  else()
    include(cmake/FindAtlas.cmake)
    if (ATLAS_FOUND)
        set(HAVE_BLAS TRUE)
        set(BLAS_INCLUDE ${Atlas_INCLUDE_DIR} )
        set(BLAS_LIBRARY ${Atlas_LIBRARIES})
    endif()
  endif()

  if (HAVE_BLAS)
    message(STATUS "Found BLAS: found (include: ${BLAS_INCLUDE}, library: ${BLAS_LIBRARY})")
    set(HAVE_BLAS ${HAVE_BLAS}  PARENT_SCOPE )
    set(BLAS_INCLUDE ${BLAS_INCLUDE} PARENT_SCOPE)
    set(BLAS_LIBRARY ${BLAS_LIBRARY} PARENT_SCOPE)
  endif()

endfunction()

##########################################################################################
function(detect_all)
  if (NOT CPU_ONLY)
    detect_cuda()
    if (HAVE_CUDA)
      list(APPEND PROJECT_INCLUDE ${CUDA_INCLUDE})
      list(APPEND PROJECT_LIBRARY ${CUDA_LIBRARY})
    endif()

    detect_cuDNN()
    if (HAVE_CUDNN)
      list(APPEND PROJECT_INCLUDE ${CUDNN_INCLUDE})
      list(APPEND PROJECT_LIBRARY ${CUDNN_LIBRARY})
    endif()
  else()
    detect_blas()
    if (HAVE_BLAS)
      list(APPEND PROJECT_INCLUDE ${BLAS_INCLUDE})
      list(APPEND PROJECT_LIBRARY ${BLAS_LIBRARY})
    endif()
  endif()


  detect_protobuf()
  if (HAVE_PROTOBUF)
    list(APPEND PROJECT_INCLUDE ${PROTOBUF_INCLUDE})
    list(APPEND PROJECT_LIBRARY ${PROTOBUF_LIBRARY})
  endif()

  detect_boost()
  if (HAVE_BOOST)
    list(APPEND PROJECT_INCLUDE ${BOOST_INCLUDE})
    list(APPEND PROJECT_LIBRARY ${BOOST_LIBRARY})
  endif()

  detect_tbb()
  if (HAVE_TBB)
    list(APPEND PROJECT_INCLUDE ${TBB_INCLUDE})
    list(APPEND PROJECT_LIBRARY ${TBB_LIBRARY})
  endif()

  detect_opencv()
  if (HAVE_OPENCV)
    list(APPEND PROJECT_INCLUDE ${OPENCV_INCLUDE})
    list(APPEND PROJECT_LIBRARY ${OPENCV_LIBRARY})
  endif()

  list(REMOVE_DUPLICATES PROJECT_INCLUDE)
  list(REMOVE_DUPLICATES PROJECT_LIBRARY)

  set(PROJECT_INCLUDE ${PROJECT_INCLUDE} PARENT_SCOPE)
  set(PROJECT_LIBRARY ${PROJECT_LIBRARY} PARENT_SCOPE)

endfunction()
