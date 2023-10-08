#ifndef __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_PERMUTE_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class PermuteLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
     std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
     void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			       std::vector<int>& permute_orders,
			       std::string& name);

  }; // class PermuteLayerBuilder

  REGISTER_LAYER_BUILDER(PermuteLayerBuilder, Permute);
} // namespace facethink

#endif
