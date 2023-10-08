link_directories(${PROJECT_BINARY_DIR}/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(blob_test blob_test.cpp)
target_link_libraries(blob_test ${LIBRARY_NAME}) 

add_executable(ssd_test_${PLATFORM_TYPE} ssd_test.cpp)
target_link_libraries(ssd_test_${PLATFORM_TYPE} ${LIBRARY_NAME})

add_executable(ssd_shuffle_test_${PLATFORM_TYPE} ssd_shuffle_test.cpp)
target_link_libraries(ssd_shuffle_test_${PLATFORM_TYPE} ${LIBRARY_NAME}) 

add_executable(faster_rcnn_net_test_${PLATFORM_TYPE} faster_rcnn_test.cpp)
target_link_libraries(faster_rcnn_net_test_${PLATFORM_TYPE} ${LIBRARY_NAME})

add_executable(face_angle_test_${PLATFORM_TYPE} face_angle_test.cpp)
target_link_libraries(face_angle_test_${PLATFORM_TYPE} ${LIBRARY_NAME}) 

add_executable(face_det_test_${PLATFORM_TYPE} face_det_test.cpp)
target_link_libraries(face_det_test_${PLATFORM_TYPE} ${LIBRARY_NAME})

add_executable(landmark_net_test_${PLATFORM_TYPE} landmark_net_test.cpp)
target_link_libraries(landmark_net_test_${PLATFORM_TYPE} ${LIBRARY_NAME}) 

add_executable(ict_ssd_test_${PLATFORM_TYPE} ict_ssd_test.cpp)
target_link_libraries(ict_ssd_test_${PLATFORM_TYPE} ${LIBRARY_NAME}) 
