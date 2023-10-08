file(GLOB_RECURSE rest *.cc *.cpp)
file(GLOB_RECURSE cuda *.cu)
file(GLOB_RECURSE gpu *_cudnn.cpp *_cuda.cpp *_gpu.cpp)
file(GLOB_RECURSE cpu *_cpu.cpp)

list(REMOVE_ITEM rest ${gpu} ${cpu})

if (WIN32)
  if (NOT CPU_ONLY)
    # cuda_compile(cuda_objs STATIC ${cuda}) #lib
    cuda_compile(cuda_objs SHARED ${cuda})
    list(APPEND srcs ${rest} ${gpu} ${cuda_objs} ${cuda})
  else()
    list(APPEND srcs ${rest} ${cpu})
  endif()

  # add_library(${LIBRARY_NAME} STATIC ${srcs}) #lib
  add_library(${LIBRARY_NAME} SHARED ${srcs})
  target_link_libraries(${LIBRARY_NAME} ${PROJECT_LIBRARY})

else()
  if (NOT CPU_ONLY)
    # cuda_compile(cuda_objs STATIC ${cuda}) #lib
	  cuda_compile(cuda_objs SHARED ${cuda})
    list(APPEND srcs ${rest} ${gpu} ${cuda_objs} ${cuda})
  else()
    list(APPEND srcs ${rest} ${cpu})
  endif()

  # add_library(${LIBRARY_NAME} STATIC ${srcs}) #lib
  add_library(${LIBRARY_NAME} SHARED ${srcs})
  find_package( Threads REQUIRED )
  target_link_libraries(${LIBRARY_NAME} ${PROJECT_LIBRARY} Threads::Threads)

  #LIB-version  API-version
  #SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES VERSION 1.0.2 SOVERSION 2)
  if(DEFINED VERSION)
    SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES VERSION ${VERSION})
  endif()

endif()
