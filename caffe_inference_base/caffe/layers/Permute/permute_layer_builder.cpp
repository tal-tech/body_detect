#include "caffe/layers/Permute/permute_layer_builder.hpp"

#ifndef CPU_ONLY
#include "caffe/layers/Permute/permute_layer_cuda.hpp"
#else
#include "caffe/layers/Permute/permute_layer_cpu.hpp"
#endif

namespace facethink {

  template <typename Dtype>
  std::shared_ptr<BaseLayer<Dtype> >
  PermuteLayerBuilder<Dtype>::Create(const caffe::LayerParameter& layer_param) {
    std::string name;
    std::vector<int> permute_orders;
    
    ParseLayerParameters(layer_param,
			 permute_orders,
			 name);

    std::shared_ptr<BaseLayer<Dtype> > layer;
#ifndef CPU_ONLY
    layer = std::make_shared<CUDAPermuteLayer<Dtype> >(permute_orders,
						       name);
   
#else
    layer = std::make_shared<PermuteLayer<Dtype> >(permute_orders,
						   name);
#endif
    return layer;
  }

  template <typename Dtype>
  void PermuteLayerBuilder<Dtype>::ParseLayerParameters(const caffe::LayerParameter& layer_param,
							std::vector<int>& permute_orders,
							std::string& name) {
    this->ParseLayerName(layer_param, name);
    
    const caffe::PermuteParameter& permute_param = layer_param.permute_param();
    int order_size = permute_param.order_size();
    permute_orders.resize(order_size);
    for (int i = 0; i < order_size; ++i){
      permute_orders[i] = permute_param.order(i);
    }
  }
 
} // namespace facethink

