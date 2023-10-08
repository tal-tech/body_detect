#include "common_test.hpp"

#include <boost/log/utility/setup/file.hpp>
using namespace facethink;

int main() {

#ifndef CPU_ONLY  // GPU Caffe.
 int GPU_ID = 0;
	//Caffe::set_mode(Caffe::GPU);
	Caffe::DeviceQuery();
	bool id_flag = Caffe::CheckDevice(GPU_ID);
	if (id_flag) {
		Caffe::SetDevice(GPU_ID);
		std::cout << "set GPU_ID: " << GPU_ID << std::endl;
	}
#endif


	boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);

  const std::string prototxt = "../../data/face_det/det_face_1.prototxt";
  const std::string model_file = "../../data/face_det/det_face_1.caffemodel";

  const std::string file = "../../data/face_det/det_face_1.bin";

  facethink::facethinkcaffe::NetBuilder<float> net_builder;
  auto net = net_builder.Create(prototxt, model_file);
  //auto net = net_builder.Create(file);

  std::cout<<"################# Load Pycaffe data just for verification test ################"<<std::endl;
  std::fstream fin("../../data/face_det/pydata_3_384_216", std::ios::in);

  std::vector<float> fdata;
  float ff;
  while(fin >> ff){
    fdata.push_back(ff);
  }
  fin.close();

	int flag = 0;
	while(1){
  std::vector<int> shape = { 1, 3, 384, 216 };
  //std::vector<int> shape = { 1, 3, 224, 224 };

  auto blob_data = net->blob("data");
  blob_data->Reshape(shape);
  blob_data->ImportFromExtenalData(fdata.data(), 3 * 1 * 384 * 216);
  //blob_data->ImportFromExtenalData(fdata.data(), 3 * 1 * 224 * 224);

  net->Forward();

  //PrintBlob(net->blob("prob1"), 1 * 2 * 187 * 103);
  //PrintBlob(net->blob("prob1"), 100);
  //PrintBlob(net->blob("poselayer"), 100);
	if (flag % 500 ==0) {
		std::cout<< "flag: " <<flag << std::endl;
	}

	flag++;
}
  std::getchar();
  return 0;
}
