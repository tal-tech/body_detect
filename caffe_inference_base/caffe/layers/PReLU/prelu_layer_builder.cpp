#include "caffe/layers/PReLU/prelu_layer_builder.hpp"
#include "caffe/util/blob_util.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/PReLU/prelu_layer_cuda.hpp"
#else
#include "caffe/layers/PReLU/prelu_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  PReluLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    bool inplace;
	bool channel_shared;
    
    ParseLayerParameters(layer_param, inplace, channel_shared, name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAPReluLayer<Dtype> >(inplace, channel_shared, name);
#else			
    layer = std::make_shared<PReluLayer<Dtype> >(inplace, channel_shared, name);
#endif
    return layer;
  }

  template<typename Dtype>
  void PReluLayerBuilder<Dtype>::ImportData(const caffe::LayerParameter& layer_param, std::shared_ptr<BaseLayer<Dtype> >& layer) {

	  std::shared_ptr<BasePReluLayer<Dtype> > prelu_layer = std::static_pointer_cast<BasePReluLayer<Dtype> >(layer);

	  if (IsShapeEqual(layer_param, prelu_layer)) {
		  std::vector<std::shared_ptr<Blob<Dtype> > > weights;

		  const int blob_size = layer_param.blobs_size();
		  for (int i = 0; i < blob_size; ++i) {
			  std::shared_ptr<Blob<Dtype> > weight = std::make_shared<Blob<Dtype> >();
			  BlobUtil<Dtype>::ImportFromProto(layer_param.blobs(i), weight);
			  weights.push_back(weight);
		  }

		  prelu_layer->SetWeights(weights);

		  if (weights.size() == 1) {
			  BOOST_LOG_TRIVIAL(info) << "PReluLayerBuilder: Layer(" << prelu_layer->name() << ") import"
				  << " bias: " << weights[0]->shape_string();
		  }
		  else {
			  BOOST_LOG_TRIVIAL(info) << "PReluLayerBuilder: Layer(" << prelu_layer->name() << ") import"
				  << " scale: " << weights[0]->shape_string()
				  << ", bias: " << weights[1]->shape_string();
		  }

	  }
	  else {
		  BOOST_LOG_TRIVIAL(error) << "PReluLayerBuilder: LayerParameter not match.";
	  }
  }


  template <typename Dtype>
  bool PReluLayerBuilder<Dtype>::IsShapeEqual(const caffe::LayerParameter& layer_param, std::shared_ptr<BasePReluLayer<Dtype> >& layer) {
	  std::string name;
	  bool inplace;
	  bool channel_shared;

	  ParseLayerParameters(layer_param, inplace, channel_shared, name);

	  return (layer->name() == name      &&
		  layer->inplace() == inplace     &&
		  layer->channel_shared() == channel_shared);
  }

  template <typename Dtype>
  void PReluLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						     bool& inplace,
						     bool& channel_shared,
						     std::string& name) {
    this->ParseLayerName(layer_param, name);

    inplace = false;
    if (!layer_param.top(0).compare(layer_param.bottom(0))) {
      inplace = true;
    }

    const caffe::PReLUParameter& prelu_param = layer_param.prelu_param();

	channel_shared = false;
    if (prelu_param.has_channel_shared()) {
		channel_shared = prelu_param.channel_shared();
    }
  }

} // namespace facethink
