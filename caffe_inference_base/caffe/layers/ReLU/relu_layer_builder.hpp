#ifndef __FACETHINK_CAFFE_LAYERS_RELU_LAYER_BUILDR_HPP__
#define __FACETHINK_CAFFE_LAYERS_RELU_LAYER_BUILDR_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class ReluLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
    std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);

  private:
    void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			      bool& inplace,
			      Dtype& negative_slope,
			      std::string& name);

  }; // class ReluLayerBuilder

  REGISTER_LAYER_BUILDER(ReluLayerBuilder, ReLU);

} // namespace facethink

#endif
