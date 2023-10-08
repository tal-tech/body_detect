#ifndef __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_BUILDER_HPP__
#define __FACETHINK_CAFFE_LAYERS_DROPOUT_LAYER_BUILDER_HPP__

#include "caffe/core/base_layer_builder.hpp"
#include "caffe/core/layer_factory.hpp"

namespace facethink {

  template <typename Dtype>
  class DropoutLayerBuilder: public BaseLayerBuilder<Dtype> {
  public:
     std::shared_ptr<BaseLayer<Dtype> > Create(const caffe::LayerParameter& layer_param);
  private:
     void ParseLayerParameters(const caffe::LayerParameter& layer_param,
			       Dtype& dropout_ratio,
			       std::string& name);

  }; // class DropoutLayerBuilder

  REGISTER_LAYER_BUILDER(DropoutLayerBuilder, Dropout);

} // namespace facethink


#endif
