#include "caffe/layers/net_builder.hpp"
#include "caffe/proto/proto_io.hpp"
#include "caffe/core/layer_factory.hpp"
#include "caffe/util/blob_util.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/trivial.hpp>

#ifndef CPU_ONLY
#define LOG_NAME "CaffeNet-GPU.log"
#else
#define LOG_NAME "CaffeNet-CPU.log"
#endif

namespace facethink {
  namespace facethinkcaffe{

	template <typename Dtype>
	void NetBuilder<Dtype>::SetLog() {

		boost::log::add_file_log
		(
		 boost::log::keywords::auto_flush = true,
		 boost::log::keywords::file_name = LOG_NAME,     /*< file name pattern >*/
		 boost::log::keywords::format = "[%TimeStamp%]: %Message%"  /*< log record format >*/
	 );

		 boost::log::add_common_attributes();

		 #ifndef LOG_INFO
       boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
		 #else
		   boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::info);
		 #endif

	}

  template <typename Dtype>
  std::shared_ptr<Net<Dtype> > NetBuilder<Dtype>::Create(const std::string& proto_txt_file,
							 const std::string& proto_binary_file) {

	  //set boost log
	  SetLog();

    //net build
    std::shared_ptr<Net<Dtype> > net = std::make_shared<Net<Dtype> >();

    caffe::NetParameter net_param_txt;
    if (!facethink::ReadProtoFromTextFile(proto_txt_file, &net_param_txt)) {
      BOOST_LOG_TRIVIAL(error)<< "NetBuilder: Failed to read caffe prototxt: "<< proto_txt_file;
    }else{
      BOOST_LOG_TRIVIAL(info)<< "NetBuilder: Succeed to read caffe prototxt: "<< proto_txt_file;
    }

    ParseNetParameterText(net_param_txt, net);

    caffe::NetParameter net_param_binary;
    if (!facethink::ReadProtoFromBinaryFile(proto_binary_file, &net_param_binary)){
      BOOST_LOG_TRIVIAL(error)<<"NetBuilder: Failed to read caffe model: "<< proto_binary_file;
    }else{
      BOOST_LOG_TRIVIAL(info)<< "NetBuilder: Succeed to read caffe model: "<< proto_binary_file;
    }

    ParseNetParameterBinary(net_param_binary, net);

    return net;
  }

  template <typename Dtype>
  std::shared_ptr<Net<Dtype> > NetBuilder<Dtype>::Create(const std::string& proto_file, int key, const int len_max) {

	  //set boost log
	  SetLog();

	  //net build
	  std::shared_ptr<Net<Dtype> > net = std::make_shared<Net<Dtype> >();
	  caffe::NetParameter net_param_txt;
	  caffe::NetParameter net_param_binary;

	  bool success = facethink::ReadProtoFromModelFile(proto_file, &net_param_txt, &net_param_binary, key, len_max);
	  if (!success) {
		   BOOST_LOG_TRIVIAL(error) << "NetBuilder: Failed to read caffe model: " << proto_file;
		  }
		  else {
		   BOOST_LOG_TRIVIAL(info) << "NetBuilder: Succeed to read caffe model: " << proto_file;
		  }

	  ParseNetParameterText(net_param_txt, net);
	  ParseNetParameterBinary(net_param_binary, net);

	  return net;
  }

    template <typename Dtype>
    void NetBuilder<Dtype>::ParseNetParameterText(const caffe::NetParameter& net_param,
                                                  std::shared_ptr<Net<Dtype> >& net) {

      caffe::NetParameter v2_net_param = net_param;
      if (! caffe::UpgradeNetAsNeeded("param_file", &v2_net_param)) {
        BOOST_LOG_TRIVIAL(error) << "UpgradeNetAsNeeded Failed ";
        return;
      }

      if (v2_net_param.has_name()) {
        net->set_name(v2_net_param.name());
      }

      ParseInputBlobs(v2_net_param, net);
      LayerFactory<Dtype>& layer_factory = LayerFactory<Dtype>::instance();

      int layer_size = v2_net_param.layer_size();
      for (int i = 0; i < layer_size; ++i) {

        const ::caffe::LayerParameter& layer_param = v2_net_param.layer(i);;
        const std::string& layer_type = layer_param.type();

        if (!layer_type.compare("Input")) continue;

        InsertOutputBlobsOfLayer(layer_param, net);

        auto layer = layer_factory.get_builder(layer_type)->Create(layer_param);
        SetUpAndAppendLayer(layer_param, net, layer);

        BOOST_LOG_TRIVIAL(info) << "NetBuilder: Create " << layer->type() << "Layer " << layer->param_string();
      }
    }


    template <typename Dtype>
    void NetBuilder<Dtype>::ParseNetParameterBinary(const caffe::NetParameter& net_param,
                                                    std::shared_ptr<Net<Dtype> >& net) {
      LayerFactory<Dtype>& layer_factory = LayerFactory<Dtype>::instance();

      int layer_size = net_param.layer_size();
      for (int i = 0; i < layer_size; ++i) {
        const ::caffe::LayerParameter& layer_param = net_param.layer(i);
        if (layer_param.blobs_size() == 0)  continue;

        const std::string& layer_name = layer_param.name();
        std::shared_ptr<BaseLayer<Dtype> > target_layer = net->layer(layer_name);

        if (target_layer) {
          const std::string& layer_type = layer_param.type();
          layer_factory.get_builder(layer_type)->ImportData(layer_param, target_layer);
        }
      }
    }

    template <typename Dtype>
    void NetBuilder<Dtype>::ParseInputBlobs(const caffe::NetParameter& net_param,
                                            std::shared_ptr<Net<Dtype> >& net) {
      bool has_input_detected = false;

      int input_size = net_param.input_size();
      if (input_size > 0){
        if (input_size == net_param.input_shape_size()){
          BOOST_LOG_TRIVIAL(info)<< "NetBuilder[ParseInputBlobs]: input_shape detected.";
          for (int i=0; i<input_size; ++i){
            const std::string input_name = net_param.input(i);
            auto input_blob = std::make_shared<Blob<Dtype> >();
            BlobUtil<Dtype>::ReshapeFromProto(net_param.input_shape(i), input_blob);
            net->InsertBlob(input_name, input_blob);
            has_input_detected = true;
          }
        }else if (input_size == net_param.input_dim_size()/4){
          BOOST_LOG_TRIVIAL(info)<< "NetBuilder[ParseInputBlobs]: input_dim detected.";
          for (int i=0; i<input_size; ++i){
            const std::string input_name = net_param.input(i);
            auto input_blob = std::make_shared<Blob<Dtype> >();
            std::vector<int> input_shape = { net_param.input_dim(i*4),   net_param.input_dim(i*4+1),
                                             net_param.input_dim(i*4+2), net_param.input_dim(i*4+3) };
            input_blob->Reshape(input_shape);
            net->InsertBlob(input_name, input_blob);
            has_input_detected = true;
          }
        }
      }

      int layer_size = net_param.layer_size();
      for (int i=0; i<layer_size; ++i){
        const ::caffe::LayerParameter& layer_param = net_param.layer(i);
        const std::string layer_type = layer_param.type();
        if (!layer_type.compare("Input")){
          BOOST_LOG_TRIVIAL(info)<< "NetBuilder[ParseInputBlobs]: Input_layer detected. ";
          const caffe::InputParameter& input_param = layer_param.input_param();

          const int shape_size = input_param.shape_size();
          if (shape_size == 0){
            BOOST_LOG_TRIVIAL(error)<< "NetBuilder[ParseInputBlobs]: No shape defined in InputLayer.";
          }

          const int top_size = layer_param.top_size();
          if (shape_size > 1 && shape_size != top_size){
            BOOST_LOG_TRIVIAL(info)<< "NetBuilder[ParseInputBlobs]: The number of shape and top must be equal in InputLayer for shape_size > 1.";
          }

          for (int j=0; j<top_size; ++j){
            auto input_blob = std::make_shared<Blob<Dtype> >();
            if (shape_size == 1){
              BlobUtil<Dtype>::ReshapeFromProto(input_param.shape(0), input_blob);
            }else{
              BlobUtil<Dtype>::ReshapeFromProto(input_param.shape(j), input_blob);
            }
            const std::string input_name = layer_param.top(j);
            net->InsertBlob(input_name, input_blob);
            has_input_detected = true;
          }
        }
      }

    if (!has_input_detected){
      BOOST_LOG_TRIVIAL(fatal)<<"NetBuilder[ParseInputBlobs]: No input found in the network prototxt";
    }

  }


  template <typename Dtype>
  void NetBuilder<Dtype>::InsertOutputBlobsOfLayer(const caffe::LayerParameter& layer_param,
						   std::shared_ptr<Net<Dtype> >& net) {
    const int top_size = layer_param.top_size();
    for (int j=0; j<top_size; ++j) {
      auto top_blob = std::make_shared<Blob<Dtype> >();
      net->InsertBlob(layer_param.top(j), top_blob);
    }
  }


  template <typename Dtype>
  void NetBuilder<Dtype>::SetUpAndAppendLayer(const caffe::LayerParameter& layer_param,
					      std::shared_ptr<Net<Dtype > >& net,
					      std::shared_ptr<BaseLayer<Dtype> >& layer) {
    std::vector<std::shared_ptr<Blob<Dtype> > > inputs;
    const int bottom_size = layer_param.bottom_size();
    for (int j=0; j<bottom_size; ++j){
      inputs.push_back(net->blob(layer_param.bottom(j)));
    }

    std::vector<std::shared_ptr<Blob<Dtype> > > outputs;
    const int top_size = layer_param.top_size();
    for (int j=0; j<top_size; ++j){
      outputs.push_back(net->blob(layer_param.top(j)));
    }

    layer->SetUp(inputs, outputs);
    net->AppendLayer(layer);
  }

  INSTANTIATE_CLASS(NetBuilder);

}
} // namespace facethink
