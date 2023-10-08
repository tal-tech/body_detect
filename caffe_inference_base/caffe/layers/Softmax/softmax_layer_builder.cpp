#include "caffe/layers/Softmax/softmax_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Softmax/softmax_layer_cudnn.hpp"
#else
#include "caffe/layers/Softmax/softmax_layer_cpu.hpp"
#endif


namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  SoftmaxLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    int softmax_axis;
    
    ParseLayerParameters(layer_param,
			 softmax_axis,
			 name);
    
    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDNNSoftmaxLayer<Dtype> >(softmax_axis, name);
#else			
    layer = std::make_shared<SoftmaxLayer<Dtype> >(softmax_axis, name);
#endif
    return layer;
  }

  template <typename Dtype>
  void SoftmaxLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
						       int& softmax_axis,
						       std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::SoftmaxParameter& softmax_param = layer_param.softmax_param();

    if (softmax_param.has_axis()) {
      softmax_axis = softmax_param.axis();
    }else{
      softmax_axis = 1;
    }
  }
  
} // namespace facethink;
