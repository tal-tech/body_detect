#include "common_test.hpp"

#include <boost/log/utility/setup/file.hpp>
using namespace facethink;

int main() {

#ifndef CPU_ONLY  // GPU Caffe.
	int GPU_ID = 1;
	 //Caffe::set_mode(Caffe::GPU);
	 Caffe::DeviceQuery();
	 bool id_flag = Caffe::CheckDevice(GPU_ID);
	 if (id_flag) {
		 Caffe::SetDevice(GPU_ID);
		 std::cout << "set GPU_ID: " << GPU_ID << std::endl;
	 }
#endif


	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);

  const std::string prototxt = "../../data/face_angle/face_angle_deploy.prototxt";
  const std::string model_file = "../../data/face_angle/face_angle.caffemodel";


  facethink::facethinkcaffe::NetBuilder<float> net_builder;
  auto net = net_builder.Create(prototxt, model_file);

  std::cout<<"################# Load Pycaffe data just for verification test ################"<<std::endl;
  std::fstream fin("../../data/face_angle/single_face.txt", std::ios::in);

  std::vector<float> fdata;
  float ff;
  while(fin >> ff){
    fdata.push_back(ff);
  }
  fin.close();
	int flag = 0;
	while(1){
  std::vector<int> shape = { 1, 1, 60, 60 };

  auto blob_data = net->blob("data");
  blob_data->Reshape(shape);
  blob_data->ImportFromExtenalData(fdata.data(), 60*60);

  net->Forward();

  //PrintBlob(net->blob("Dense3"), 100);

	if (flag % 500 ==0) {
		std::cout<< "flag: " <<flag << std::endl;
	}
	flag++;
}
  std::getchar();
  return 0;
}
