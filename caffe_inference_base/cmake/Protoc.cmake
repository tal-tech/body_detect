function(caffe_protobuf_generate_cpp protoc_exe)

  set(proto_path ${PROJECT_SOURCE_DIR}/proto)
  
  execute_process(COMMAND ${protoc_exe} -I ${proto_path} ${proto_path}/caffe.proto --cpp_out=${PROJECT_SOURCE_DIR}/caffe/proto)

endfunction()
