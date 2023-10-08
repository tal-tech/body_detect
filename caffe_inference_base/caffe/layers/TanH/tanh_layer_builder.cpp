#include "caffe/layers/TanH/tanh_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/TanH/tanh_layer_cudnn.hpp"
#else
#include "caffe/layers/TanH/tanh_layer_cpu.hpp"
#endif

namespace facethink {

  
  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  TanHLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    
    ParseLayerParameters(layer_param, name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDNNTanHLayer<Dtype> >(name);
#else			
    layer = std::make_shared<TanHLayer<Dtype> >(name);
#endif
    return layer;
  }

  template <typename Dtype>
  void TanHLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
			        
						     std::string& name) {
    this->ParseLayerName(layer_param, name);
  }

} // namespace facethink
