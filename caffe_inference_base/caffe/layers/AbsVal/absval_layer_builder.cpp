#include "caffe/layers/AbsVal/absval_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/AbsVal/absval_layer_cuda.hpp"
#else
#include "caffe/layers/AbsVal/absval_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  AbsValLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    
    ParseLayerParameters(layer_param,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAAbsValLayer<Dtype> >(name);
#else
    layer = std::make_shared<AbsValLayer<Dtype> >(name);
#endif
    return layer;
  }

  template <typename Dtype>
  void AbsValLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						       std::string& name) {
    this->ParseLayerName(layer_param, name);
  }
} // namespace facethink
