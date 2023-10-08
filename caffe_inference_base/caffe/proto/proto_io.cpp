#include "caffe/proto/proto_io.hpp"

#include <boost/log/trivial.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <fcntl.h>
#include <limits>

namespace facethink {

  using google::protobuf::io::FileInputStream;
  using google::protobuf::io::ZeroCopyInputStream;
  using google::protobuf::io::CodedInputStream;

  bool ReadProtoFromTextFile(const std::string& filename, Message* proto){
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1){
      BOOST_LOG_TRIVIAL(error) << "ReadProtoFromTextFile: can not open file [" << filename<<"]";
      return false;
    }

    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
  }

  bool ReadProtoFromBinaryFile(const std::string& filename, Message* proto){
#ifdef _WIN32
    int fd = open(filename.c_str(), O_RDONLY | O_BINARY);
#else
    int fd = open(filename.c_str(), O_RDONLY);
#endif

    if (fd == -1){
      BOOST_LOG_TRIVIAL(error) << "ReadProtoFromBinaryFile: can not open file [" << filename<<"]";
      return false;
    }

    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(std::numeric_limits<int>::max(), 536870912);

    bool success = proto->ParseFromCodedStream(coded_input);

    delete coded_input;
    delete raw_input;
    close(fd);
    return success;
  }

  bool ReadProtoFromModelFile(const std::string& filename, Message* proto_txt, Message* proto_binary, int key, const int len_max) {

	  //decrypt file
	  std::string txt_string;
	  std::string binary_string;
	  FileDecrypt(filename, &txt_string, &binary_string, key, len_max);

	  //txt message
	  bool success = google::protobuf::TextFormat::ParseFromString(txt_string, proto_txt);
	  if (!success) {
		  BOOST_LOG_TRIVIAL(error) << "ReadProtoFromModelFile: can not load txt file [" << filename << "]";
		  return false;
	  }

	  //binary message
	  CodedInputStream* coded_input = new CodedInputStream((unsigned char*)binary_string.data(),binary_string.length());
	  coded_input->SetTotalBytesLimit(std::numeric_limits<int>::max(), 536870912);
	  success = proto_binary->ParseFromCodedStream(coded_input);
	  delete coded_input;

	  if (!success) {
		  BOOST_LOG_TRIVIAL(error) << "ReadProtoFromModelFile: can not load binary file [" << filename << "]";
		  return false;
	  }

	  return success;
  }


  void FileDecrypt(const std::string& proto_file, std::string* txt_string, std::string* binary_string, int key, const int len_max)
  {
	  const size_t WR_SIZE = 1024 * 1024 * 10;//每次读写的大小,此处为10M
	  FILE *p_file = NULL;
	  char c_temp;

	  p_file = fopen(proto_file.data(), "rb");//二进制读写文件
	  if (p_file == NULL)
	  {
		  BOOST_LOG_TRIVIAL(error) << "NetBuilder: Failed to read proto_file: " << proto_file;
	  }

	  fseek(p_file, 0L, 2);
	  int len_file = ftell(p_file);
	  rewind(p_file);

	  char *len_char = new char[len_max];
	  for (size_t i = 0; i < len_max; i++)
	  {
		  len_char[i] = fgetc(p_file) ^ key;
	  }
	  int len;
	  std::stringstream stream(len_char);
	  stream >> len;
	  delete[] len_char;

	  if (len > len_file || len <= 0)
	  {
		  BOOST_LOG_TRIVIAL(error) << "NetBuilder:proto_file is incomplete: " << proto_file;
	  }

	  for (size_t i = 0; i < len; i++)
	  {
		  c_temp = fgetc(p_file) ^ key;
		  *txt_string += c_temp;
	  }

	  //c_temp = fgetc(p_file) ^ key;
	  //while (!feof(p_file))//遇到文件结束 返回值为0
	  //{
	  // *binary_string += c_temp;
	  // c_temp = fgetc(p_file) ^ key;//异或运算加密文件
	  //}

	  //3-读取模型参数文件
	  long long offset_read = len_max + len;
	  long long offset_write = 0;
	  size_t num_read = 0;
	  size_t num_written = 0;
	  char *buf = new char[WR_SIZE];

	  //_fseeki64(p_file, offset_read, SEEK_SET);
    fseek(p_file, offset_read, SEEK_SET);
	  while (!feof(p_file))//遇到文件结束 返回值为0
	  {
		  //read
		  num_read = fread(buf, sizeof(char), WR_SIZE, p_file);
		  if (0 == num_read) break;
		  for (size_t i = 0; i < num_read; i++)
		  {
			  buf[i] = buf[i] ^ key;
			  *binary_string += buf[i];
		  }
		  offset_read += num_read;//文件偏移
		  //_fseeki64(p_file, offset_read, SEEK_SET);
      fseek(p_file, offset_read, SEEK_SET);

	  }
	  delete[]buf;

	  fclose(p_file);
  }

} // namespace facethink
