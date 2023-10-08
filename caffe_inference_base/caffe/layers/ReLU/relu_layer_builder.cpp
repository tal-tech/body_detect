#include "caffe/layers/ReLU/relu_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/ReLU/relu_layer_cudnn.hpp"
#else
#include "caffe/layers/ReLU/relu_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  ReluLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    bool inplace;
    Dtype negative_slope;
    
    ParseLayerParameters(layer_param, inplace, negative_slope, name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDNNReluLayer<Dtype> >(inplace, negative_slope, name);
#else			
    layer = std::make_shared<ReluLayer<Dtype> >(inplace, negative_slope, name);
#endif
    return layer;
  }

  template <typename Dtype>
  void ReluLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						     bool& inplace,
						     Dtype& negative_slope,
						     std::string& name) {
    this->ParseLayerName(layer_param, name);

    inplace = false;
    if (!layer_param.top(0).compare(layer_param.bottom(0))) {
      inplace = true;
    }

    const caffe::ReLUParameter& relu_param = layer_param.relu_param();

    negative_slope = 0;
    if (relu_param.has_negative_slope()) {
      negative_slope = static_cast<Dtype>(relu_param.negative_slope());
    }
  }

} // namespace facethink
