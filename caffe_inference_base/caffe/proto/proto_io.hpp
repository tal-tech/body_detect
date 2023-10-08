#ifndef __FACETHINK_CAFFE_PROTO_PROTO_IO_HPP__
#define __FACETHINK_CAFFE_PROTO_PROTO_IO_HPP__

#include <google/protobuf/message.h>
#include <string>
#include <sstream>  
namespace facethink {

  using google::protobuf::Message;

  bool ReadProtoFromTextFile(const std::string& filename, Message* proto);

  bool ReadProtoFromBinaryFile(const std::string& filename, Message* proto);

  bool ReadProtoFromModelFile(const std::string& filename, Message* proto_txt, Message* proto_binary, int key, const int len_max);

  void FileDecrypt(const std::string& proto_file, std::string* txt_string, std::string* binary_string, int key, const int len_max);

} // namespace facethink

#endif
