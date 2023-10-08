#include "caffe/layers/Dropout/dropout_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Dropout/dropout_layer_cuda.hpp"
#else
#include "caffe/layers/Dropout/dropout_layer_cpu.hpp"
#endif

namespace facethink {
  
  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  DropoutLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    Dtype dropout_ratio;
    
    ParseLayerParameters(layer_param,
			 dropout_ratio,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDADropoutLayer<Dtype> >(dropout_ratio,
						       name);
   
#else
    layer = std::make_shared<DropoutLayer<Dtype> >(dropout_ratio,
						   name);
#endif
    return layer;
  }

  template <typename Dtype>
  void DropoutLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							Dtype& dropout_ratio,
							std::string& name) {
    this->ParseLayerName(layer_param, name);

    const caffe::DropoutParameter dropout_param = layer_param.dropout_param();

    if (dropout_param.has_dropout_ratio()){
      dropout_ratio = dropout_param.dropout_ratio();
    }else{
      dropout_ratio = Dtype(0.5);
    }
  }
  
}
